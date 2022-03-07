"""
Copyright (c) 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor, VGG_KOR_FeatureExtractor
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'VGG_KOR':
            self.FeatureExtraction = VGG_KOR_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            print("prediction given: ", opt.Prediction)
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        if ',' in character:
            dict_character = character.split(',')
            self.multi_character = True
        else:
            dict_character = list(character)
            self.multi_character = False

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if self.multi_character:
            length = [len(s.split(',')) for s in text]
            temp = []
            for t in text:
                temp += t.split(',')
            text = temp
        else:
            length = [len(s) for s in text]
            text = ''.join(text)
        
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        def eng2kor(eng):
            e2k_dict = {"rk":"가", "sk":"나", "ek":"다", "fk":"라", "ak":"마", "qk":"바", "tk":"사", "dk":"아", "wk":"자",
                    "rj":"거", "sj":"너", "ej":"더", "fj":"러", "aj":"머", "qj":"버", "tj":"서", "dj":"어", "wj":"저", 
                    "rh":"고", "sh":"노", "eh":"도", "fh":"로", "ah":"모", "qh":"보", "th":"소", "dh":"오", "wh":"조", 
                    "rn":"구", "sn":"누", "en":"두", "fn":"루", "an":"무", "qn":"부", "tn":"수", "dn":"우", "wn":"주", 
                    "gj":"허", "gk":"하", "gh":"호", "a":"서울", "b":"경기", "c":"인천", "d":"강원", "e":"충남",
                    "f":"대전", "g":"충북", "h":"부산", "i":"울산", "j":"대구", "k":"경북", "l":"경남", "m":"전남", "n":"광주",
                    "o":"전북", "p":"제주", "세종": "세종", "임":"임", "크":"크","0":"0", "1":"1", "2":"2", "3":"3", "4":"4", "5":"5", "6":"6",
                "7":"7", "8":"8", "9":"9", "[s]":"[s]", "[GO]":"[GO]", "배":"배"}
            return e2k_dict[eng]
        texts = []
        index = 0
        for l in length:
            if type(text_index) == type([]): # gt
                for text_string in text_index:
                    char_list = []
                    for t in text_string.split(','):
                        char_list.append(eng2kor(t))
                    text = ''.join(char_list)
                    texts.append(text)
            else:
                t = text_index[index:index + l]

                char_list = []
                for i in range(l):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                        if self.multi_character:
                            char_list.append(eng2kor(self.character[t[i]]))
                        else:
                            char_list.append(self.character[t[i]])
                text = ''.join(char_list)

                texts.append(text)
                index += l
        return texts