import os, glob
import cv2, tqdm
import xml.etree.ElementTree as ET


def generate_text_file():
    target = 'Set6'
    anno_path = "/data/ocr/dyiot_lpd/project-6-at-2022-02-01-02-50-41ba2520/Annotations"
    # anno_path = "/data/ocr/dyiot_lpd/project-9-at-2022-01-22-05-26-87d961fc/Annotations"
    text_path = "/data/ocr/dyiot_lpd/ImageSets/Main"
    annos_1 = glob.glob(os.path.join(anno_path, '*.xml'))
    os.makedirs(text_path, exist_ok=True)
    with open(text_path+'/trainval_{}.txt'.format(target), "a") as f:
        for anno in annos_1:
            f.write("{}\n".format(anno.split('/')[-1].split('.')[0]))
    
    return 0

def remove_bg_imgs_from_label_studio():
    
    # anno_path = "/workspace/LPD-end-to-end/images/dyiot/Annotations"
    # img_path = "/workspace/LPD-end-to-end/images/dyiot/JPEGImages"
    root_path = "/data/ocr/dyiot_lpd"
    # anno_paths = ["project-6-at-2022-01-24-06-41-a441a05e", "project-7-at-2022-01-24-08-02-c6aa75a3"]
    # anno_paths = ["project-6-at-2022-02-01-02-50-41ba2520"]
    # img_paths = ["Set6"]
    # img_paths = ["Set6", "Set7"]
    # anno_paths = ["project-1-at-2022-01-22-06-16-423806e6", "project-2-at-2022-01-22-06-07-ff3102fb",
    #              "project-3-at-2022-01-22-06-03-03596b19", "project-4-at-2022-01-22-16-09-02a24ddd",
    #              "project-5-at-2022-01-22-06-00-d4cc5a04", "project-6-at-2022-02-01-02-50-41ba2520",
    #              "project-8-at-2022-01-22-05-26-77769d09", "project-9-at-2022-01-22-05-26-87d961fc"]
    anno_paths = ["project-7-at-2022-01-24-08-02-c6aa75a3"]
    # img_paths = ["Set1", "Set2","Set3","Set4","Set5","Set6","Set8","Set9"]
    img_paths = ["Set7"]

    for (anno_path, img_path) in zip(anno_paths, img_paths):
        anno_path = os.path.join(root_path, anno_path, 'Annotations')
        img_path = os.path.join(root_path, img_path)
        annos_1 = glob.glob(os.path.join(anno_path, '*.xml'))
        annos = []
        for anno in annos_1:
            annos.append(anno.split('/')[-1].split('.')[0])
            os.system("cp {} {}".format(anno, os.path.join(root_path, "Annotations")))
        imgs = glob.glob(os.path.join(img_path, "*.jpg"))
        for img in imgs:
            file_name = img.split('/')[-1].split('.')[0]
            img_file = os.path.join(img_path, file_name+".jpg")
            if file_name not in annos:
                os.system("rm {}".format(img_file))
                print("{} was removed\n".format(file_name+'.jpg'))
            else:
                os.system("cp {} {}".format(img_file, os.path.join(root_path, "JPEGImages")))
                
    return 0
def check_labels():
    root_path = "/data/ocr/dyiot_lpd"
    img_path = os.path.join(root_path, "JPEGImages")
    anno_path = os.path.join(root_path, "Annotations")
    l_imgs = glob.glob(os.path.join(img_path, "*.jpg"))
    l_annos = glob.glob(os.path.join(anno_path, "*.xml"))
    # for i, (img, anno) in enumerate(zip(l_imgs, l_annos)):
    for anno in range(29000, len(l_annos)):
        if i % 200 == 0:
            print("[{}/{}] completed\n".format(i, len(l_annos)))
        # im = cv2.imread(img)
        # try:
        # height, width, _ = im.shape
        target = ET.parse(anno).getroot()
        # except:
        #     print("img_path:{}\n".format(img))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
 
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # if len(bndbox) == 0 or height == 0 or width == 0:
            if len(bndbox) == 0:
                print("bndbox: {}\n".format(bndbox))
                # print("bndbox: {}\timg_path:{}".format(bndbox, img))

if __name__ == '__main__':
    remove_bg_imgs_from_label_studio()
    # generate_text_file()
    # check_labels()