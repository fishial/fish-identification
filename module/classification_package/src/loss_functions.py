import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners, trainers, samplers
from pytorch_metric_learning.samplers import MPerClassSampler

class CombinedLoss(nn.Module):
    """
    A weighted combination of a classification loss (CrossEntropy on ArcFace logits)
    and a metric learning loss.
    """
    def __init__(self, arcface_weight=0.7, metric_weight=0.3):
        super().__init__()
        self.arcface_criterion = nn.CrossEntropyLoss()

        # Using ThresholdConsistentMarginLoss for the metric learning component
        self.metric_loss = losses.ThresholdConsistentMarginLoss()

        self.miner = miners.BatchHardMiner()  # You can experiment with the miner later
        self.arcface_weight = arcface_weight
        self.metric_weight = metric_weight

    def forward(self, embeddings, arcface_logits, labels):
        # Classification loss
        arcface_loss = self.arcface_criterion(arcface_logits, labels)

        # Metric learning loss
        hard_pairs = self.miner(embeddings, labels)
        metric_loss = self.metric_loss(embeddings, labels, indices_tuple=hard_pairs)

        # Combine the two losses
        total_loss = self.arcface_weight * arcface_loss + self.metric_weight * metric_loss
        return total_loss
    
    
class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class WrapperOHNM(nn.Module):
    def __init__(self):
        super(WrapperOHNM, self).__init__()
        
        self.p = 2
        self.margin = 0.1
        self.eps = 1e-7
        
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="all")
      
    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        
        hard_triplets = self.miner(feats, labels)
        loss = self.loss_func(feats, labels, hard_triplets)
        return loss
    

class WrapperAngular(nn.Module):
    def __init__(self):
        super(WrapperAngular, self).__init__()
        
     
        self.loss_func = losses.AngularLoss()
        self.miner = miners.AngularMiner()
      
    def forward(self, feats, labels):

        hard_triplets = self.miner(feats, labels)
        print(f"hard_triplets: {len(hard_triplets)} {hard_triplets[0].shape}")
        loss = self.loss_func(feats, labels, hard_triplets)
        return loss
    
    
class WrapperPNPLoss(nn.Module):
    def __init__(self):
        super(WrapperPNPLoss, self).__init__()
        
     
        self.loss_func = losses.PNPLoss()
      
    def forward(self, feats, labels):

        loss = self.loss_func(feats, labels)
        return loss
    
    
    
class QuadrupletLoss(object):

    def __init__(self, adaptive_margin=True):
        super().__init__()
        self.adaptive_margin = adaptive_margin

    def __call__(self, embeddings, labels, margin=1):
        """Build the quadruplet and triplet loss over a batch of embeddings.

        I generate all the valid quadruplet and average the loss over the positive ones.

        Args:
            labels: `Tensor` labels of the batch, of size (batch_size,)
            embeddings: `Tensor` tensor of shape (batch_size, embed_dim)
            margin: `Float` margin for quadruplet loss
            adaptive_margin: `Boolean` if set  will be used adaptive calculation otherwise there will be a constant

        Returns:
            quadruplet_loss: scalar tensor containing the triplet loss
        """

        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist
        quadruplet_loss = triplet_loss.unsqueeze(3) + triplet_loss.unsqueeze(2)

        # Put to zero the invalid triplets and quadruplet
        mask_triplet, mask_quadruplet, i_equal_j, i_not_equal_k = self._get_losses_mask(labels)

        # Adaptive margin is calculated according to the formula
        # alpha_1,2 = w(1/N_n sum_(i,k)^n g(x_i, x_k)^2 - 1/N_p sum_(i,j)^n g(x_i, x_j)^2);
        # w = 1.0 for alpha_1 and w = 0.5 for alpha_2

        if self.adaptive_margin:
            ap_sum = i_equal_j.sum()
            an_sum = i_not_equal_k.sum()
            i_equal_j_distance = i_equal_j.float() * pairwise_dist
            i_not_equal_k_distance = i_not_equal_k.float() * pairwise_dist
            margin = i_not_equal_k_distance.sum() / (an_sum + 1e-16) - i_equal_j_distance.sum() / (ap_sum + 1e-16)

            alpha_1_2 = F.relu(1.5 * margin)
            quadruplet_loss = quadruplet_loss + alpha_1_2

        else:
            triplet_loss = triplet_loss + margin
            quadruplet_loss = quadruplet_loss + margin

        # Get the loss matrix for triplet
        triplet_loss = mask_triplet.float() * triplet_loss
        # Get the loss matrix for quadruplet
        quadruplet_loss = mask_quadruplet.float() * quadruplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0
        quadruplet_loss[quadruplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_qudruplets = quadruplet_loss[quadruplet_loss > 1e-16]
        num_positive_qudruplets = valid_qudruplets.size(0)

        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)

        num_valid_triplets = mask_triplet.sum()
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        num_valid_qudruplets = mask_quadruplet.sum()
        fraction_positive_qudruplets = num_positive_qudruplets / (num_valid_qudruplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        # Get final mean quadruplet loss over the positive valid triplets
        quadruplet_loss = quadruplet_loss.sum() / (num_positive_qudruplets + 1e-16)

        return quadruplet_loss

    def _get_losses_mask(self, labels):
        """Return a 4D mask
        A quadruplet (i, j, k, l) is valid if:
            - i, j, k, l are distinct
            - labels[i] == labels[j] and labels[i] != labels[k] and labels[l] != (labels[i] and labels[j] and labels[k])
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0)).bool().cuda()
        indices_not_equal = ~indices_equal

        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        # Creation of a 4D matrix that excludes equal indices
        quadruplet_mask = distinct_indices.unsqueeze(1) & distinct_indices.unsqueeze(2)

        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = (~i_equal_k & i_equal_j).cuda()

        # Apply two conditions
        triplet_final_mask = valid_labels.cuda() & distinct_indices.cuda()
        quadruplet_final_mask = (valid_labels.unsqueeze(2) & valid_labels.unsqueeze(3)) & quadruplet_mask.cuda()

        # Mask of indices which label[i] != label[k]
        i_equal_j = (label_equal & indices_not_equal)
        # Mask of indices which label[i] == label[j]
        i_not_equal_k = (~label_equal & indices_not_equal)

        return triplet_final_mask, quadruplet_final_mask, i_equal_j, i_not_equal_k

    def _pairwise_distances(self, embeddings):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 - mask) * torch.sqrt(distances)

        return distances


def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(
        torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances


def TripletSemiHardLoss(y_true, y_pred, device, margin=10.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + \
                      axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + \
                      axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'

    def forward(self, embeddings, labels, **kwargs):
        return TripletSemiHardLoss(labels, embeddings, self.device)