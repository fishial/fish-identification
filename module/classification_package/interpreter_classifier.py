import torch.nn as nn
import numpy as np
import logging
import torch

from PIL import Image
from torchvision import transforms

class EmbeddingClassifier:
    def __init__(self, model_path, data_set_path, indexes_of_elements, device='cpu', THRESHOLD = 6.84):
        self.device = device
        self.THRESHOLD = THRESHOLD
        self.map_of_items = indexes_of_elements['list_of_ids']
        self.categories = indexes_of_elements['categories']
        self.softmax = nn.Softmax(dim=1)
        
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(device)
        
        self.loader = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.data_base = torch.load(data_set_path).to(device)
        logging.info("[INIT][CLASSIFICATION] Initialization of classifier was finished")
                
    def __inference(self, image, top_k = 15): 
        logging.info("[PROCESSING][CLASSIFICATION] Getting embedding for a single detection mask")
        
        dump_embed, fc_output = self.model(image.unsqueeze(0).to(self.device))
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by Full Connected layer for a single detection mask")  
        classes, _ = self.__classify_fc(fc_output)
        
        fc_recognized = self.categories[str(classes[0].item())]
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
        output_by_embeddings = self.__classify_embedding(dump_embed[0], top_k)
        
        logging.info("[PROCESSING][CLASSIFICATION] Beautify output for a single detection mask")
        result = self.__beautifier_output(output_by_embeddings, fc_recognized)
        return result
                        
    def __beautifier_output(self, output_by_embeddings, classification_label):
        dict_results = []
        already_writed = []
        
        for class_map in output_by_embeddings:
            if class_map['name'] not in already_writed:
                dict_results.append(class_map)
                already_writed.append(class_map['name'])
            else:
                pass
                
        if classification_label['name'] not in already_writed:
            logging.info("[PROCESSING][CLASSIFICATION] Append into output classification result by FC - layer")
            dict_results.append(
                {
                    'name': classification_label['name'],
                    'species_id': classification_label['species_id'],
                    'distance': self.THRESHOLD,
                    'accuracy': 0.01,
                    'image_id': None,
                    'annotation_id': None,
                    'drawn_fish_id': None,
                }
                
            )
        return dict_results
    
    def __get_confidence(self, dist):
        min_dist = 3.5
        max_dist = self.THRESHOLD
        delta = max_dist - min_dist
        return 1.0 - (max(min(max_dist, dist), min_dist) - min_dist) / delta
    
    def inference_numpy(self, img, top_k=10):
        image = Image.fromarray(img)
        image = self.loader(image)
        
        return self.__inference(image, top_k)
    
    def batch_inference(self, imgs):
        batch_input = []
        for idx in range(len(imgs)):
            image = Image.fromarray(imgs[idx])
            image = self.loader(image)
            batch_input.append(image)

        batch_input = torch.stack(batch_input)
        dump_embeds, class_ids = self.model(batch_input)
        
        logging.info("[PROCESSING][CLASSIFICATION] Classification by Full Connected layer for a single detection mask")  
        classes, scores = self.__classify_fc(class_ids)
       
        outputs = []
        for output_id in range(len(classes)):

            logging.info("[PROCESSING][CLASSIFICATION] Classification by embedding for a single detection mask")
            output_by_embeddings = self.__classify_embedding(dump_embeds[output_id])
            result = self.__beautifier_output(output_by_embeddings, self.categories[str(classes[output_id].item())])
            outputs.append(result)
        return outputs
    
    def __classify_fc(self, output):
        acc_values = self.softmax(output)
        class_id = torch.argmax(acc_values, dim=1)
        #print(f"Recognized species id {class_id} with liklyhood: {acc_values[0][class_id]}")
        return class_id, acc_values

    def __classify_embedding(self, embedding, top_k = 15):
        diff = (self.data_base - embedding).pow(2).sum(dim=1).sqrt()
        val, indi = torch.sort(diff)
        
        embedding_classification_output = []
        for indiece in indi[:top_k]:
            internal_id, image_id, annotation_id, drawn_fish_id = \
            self.map_of_items[indiece]
                   
            class_info_map  = {
                'name': self.categories[str(internal_id)]['name'],
                'species_id': self.categories[str(internal_id)]['species_id'],
                'distance': diff[indiece].item(),
                'accuracy': round(self.__get_confidence(diff[indiece].item()), 3),
                'image_id': image_id,
                'annotation_id': annotation_id,
                'drawn_fish_id': drawn_fish_id,
            }
            embedding_classification_output.append(class_info_map)
        return embedding_classification_output

    
    def __get_species_name(self, category_name):
        for i in self.categories:
            if self.categories[i]['name'] == category_name:
                return self.categories[i]['species_id']