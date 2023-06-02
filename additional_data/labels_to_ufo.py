import json
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--folder_path', type=str, default= './annotations')
    parser.add_argument('--output_dir', type=str, default= './ufo')
    
    args = parser.parse_args()

    return args

def extract_info(folder_path, file_name): #하나의 이미지에 대한 json 파일에서 annotation 정보를 추출
        with open(folder_path+'/'+file_name, encoding="UTF8") as json_reader:
            try:
                data = json.load(json_reader)
            except json.decoder.JSONDecodeError:
                print("data is not a valid JSON string", file_name)
                return "no file"
            
        words = data['annotations'][0]['polygons']
        words_info = {}
        for w in words:            
            w_info = {
                        w["id"]: {
                        "transcription": w["text"], 
                        "points": w["points"], 
                        "orientation": "Horizontal",
                        "language": None,
                        "tags": [
                            "Auto"
                        ],
                        "confidence": None,
                        "illegibility": False
                        }
                    }
            words_info.update(w_info)
        
        img = data['images'][0]
        img_info = { img['name']: {
            "paragraphs": {}, 
            "words": words_info
            }
        }
        return img_info

def merge_labels(folder_path, output_dir): #이미지별 label을 하나의 .json 파일로 변경
    file_list = os.listdir(folder_path) 
    images_info = {}
    
    for json_file in file_list:
        img_info = extract_info(folder_path, json_file)
        if img_info != "no file":
            images_info.update(img_info)

    os.makedirs(output_dir, exist_ok=True)
    output_json_filename = os.path.join(output_dir, 'label.json')

    label_json = {"images": images_info}
    with open(output_json_filename, 'w', encoding="UTF8") as train_writer:
        json.dump(label_json, train_writer, indent=4)


def main(args):
    merge_labels(args.folder_path, args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)