import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

parser = argparse.ArgumentParser(description="View images with bboxes from the COCO dataset")
parser.add_argument("-a", "--annotations", default='', type=str, metavar="PATH", help="path to annotations json file")


def read_json(data):
    with open(data) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def print_info(message: str):
    logging.info(message)


def valid_mask(single_mask, valid_cat):
    warning_list = []
    body = True

    if 'category_id' not in single_mask:
        body = False
        warning_list.append("['annotations']['category_id'] isn't exist")
        return body, warning_list
    else:
        if single_mask['category_id'] not in valid_cat:
            body = False
            return body, warning_list
    if 'bbox' not in single_mask:
        warning_list.append("['annotations']['bbox'] isn't exist")

    if min(single_mask['bbox']) < 0:
        warning_list.append("['annotations']['bbox'] negative value")
    if len(single_mask['segmentation'][0]) < 10:
        warning_list.append("['annotations']['segmentation'] too small points count")
    return body, warning_list


def get_correct_category_ids(export):
    categories = []

    for category in export['categories']:
        if category['name'] == 'General body shape':
            categories.append(category['id'])
    return categories


def valid_image(image):
    warning_list = []
    # critical = False
    # if 'id' not in image:
    #     critical = True
    #     warning_list.append("['images']['id'] isn't exist")
    if 'width' not in image or 'height' not in image:
        warning_list.append("['images']['width'] or ['images']['height'] isn't exist")
    else:
        if type(image['width']) != int or type(image['height']) != int:
            warning_list.append(f"['images']['width'] or ['images']['height'] wrong type")
    if 'fishial_extra' not in image:
        warning_list.append("['images']['fishial_extra'] isn't exist")
    else:
        if 'test_image' not in image['fishial_extra']:
            warning_list.append("['images']['fishial_extra']['test_image'] isn't exist")
        else:
            if not type(image['fishial_extra']['test_image']) is bool:
                warning_list.append("['images']['fishial_extra']['test_image'] wrong type")
    # if "file_name" not in image and "coco_url" not in image:
    #     warning_list.append("['images']['fishial_extra']['file_name'] isn't exist")
    return warning_list


def get_invalid(export, category_list):
    data_warnings = {}

    for cr_id, image_class in enumerate(export['images']):
        print(f"Left: {len(export['images']) - cr_id}", end='\r')
        warn_list = valid_image(image_class)
        if len(warn_list) > 0:
            data_warnings.update({
                image_class['id']: {
                    'image_warn': warn_list
                }
            })

        for ann_id, ann in enumerate(export['annotations']):
            if ann['image_id'] == image_class['id']:
                infos = valid_mask(ann, category_list)

                if infos[0]:
                    if len(infos[1]) > 0:
                        if image_class['id'] in data_warnings:
                            if 'annotations_warn' in data_warnings[image_class['id']]:
                                data_warnings[image_class['id']]['annotations_warn'].append({ann_id: infos[1]})
                            else:
                                data_warnings.update({
                                    image_class['id']: {
                                        'annotations_warn': [{ann_id: infos[1]}]
                                    }
                                })
                        else:
                            data_warnings.update({
                                image_class['id']: {
                                    'annotations_warn': [{ann_id: infos[1]}]
                                }
                            })
    return data_warnings


def main():
    print_info("Starting...")
    args = parser.parse_args()

    export_dict = read_json(args.annotations)
    category_list = get_correct_category_ids(export_dict)
    data_warnings = get_invalid(export_dict, category_list)
    save_json(data_warnings, "../warnings.json")


if __name__ == "__main__":
    main()