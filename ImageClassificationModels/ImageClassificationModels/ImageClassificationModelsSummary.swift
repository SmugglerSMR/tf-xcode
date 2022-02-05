//
//  ImageClassificationModelsRead.swift
//  ImageClassificationModels
//
//  Created by Рамиль Садыков on 19.06.2020.
//  Copyright © 2020 Sadukow. All rights reserved.
//

import Foundation

import ModelSupport
import TensorFlow


protocol SummaryLayer {
    func summary(scope: String)
}


extension ZeroPadding2D: SummaryLayer {
    func summary(scope: String) {
        print("ZeroPadding2D                                                          \(scope)")
        print("_______________________________________________________________________________________________________")
    }
}

extension BatchNorm: SummaryLayer {
    func summary(scope: String) {
        //print("BatchNorm.summary    \(scope)")
        let x1 = String(format: "[ %7d ]", self.offset.shape[0])
        let x2 = String(format: " %7d ", self.scale.shape[0])
        print("BatchNorm                       \(x1)              \(x2)     \(scope)")
        print("_______________________________________________________________________________________________________")
        //print(self.offset.shape)
        //print(self.scale.shape)
        //print(self.momentum)
        //print(self.epsilon)
    }
}






extension GlobalAvgPool2D: SummaryLayer {
    func summary(scope: String) {
        print("GlobalAvgPool2D                                                        \(scope)")
        print("_______________________________________________________________________________________________________")
    }
}


extension DepthwiseConv2D:SummaryLayer {
    func summary(scope: String) {
        let x1 = String(format: "[ %3d %3d %5d  %5d ]", self.filter.shape[0], self.filter.shape[1], self.filter.shape[2], self.filter.shape[3])
        let x2 = String(format: " %5d ", self.bias.shape[0])
        print("DepthwiseConv2D                 \(x1)   \(x2)     \(scope)")
        print("_______________________________________________________________________________________________________")
    }
}


extension Conv2D: SummaryLayer {
    func summary(scope: String) {
        //print("Conv2D.summary    \(scope)")
        let x = self.filter.shape
        let x1 = String(format: "[ %3d %3d %5d  %5d ]", x[0], x[1], x[2], x[3])
        let x2 = String(format: " %5d ", self.bias.shape[0])
        print("Conv2D                          \(x1)   \(x2)     \(scope)")
        print("_______________________________________________________________________________________________________")
        //print(self.bias.shape)
        //print(self.dilations)
        //print(self.strides)
        //print(self.padding)
        //let duration = String(format: "%.01f", 3.32323242)
    }
}

extension LayerNorm: SummaryLayer {
    func summary(scope: String) {
        print("LayerNorm.summary    \(scope)")
    }
}

extension Flatten: SummaryLayer {
    func summary(scope: String) {
        print("Flatten                                     \(scope)")
        print("_______________________________________________________________________________________________________")
    }
}

extension Dense: SummaryLayer {
    func summary(scope: String) {
        let x1 = String(format: "[ %5d  %5d ]", self.weight.shape[0], self.weight.shape[1])
        let x2 = String(format: " %5d ", self.bias.shape[0])
        print("Dense                           \(x1)           \(x2)     \(scope)")
        print("_______________________________________________________________________________________________________")
    }
}

public func ReLU_summary(scope: String) {
    print("ReLU                                                                   \(scope)")
    print("_______________________________________________________________________________________________________")
}

public func Add_summary(scope: String) {
    print("Add                                                                    \(scope)")
    print("_______________________________________________________________________________________________________")
}


public func ImageClassificationSummary(model: Any, name: String) throws {
    
    print("_______________________________________________________________________________________________________")
    print("Layer (type)                    Output Shape               Param #     Connected to                     ")
    print("=======================================================================================================")
    
    if let SummaryLayer = model as? SummaryLayer {
        SummaryLayer.summary(scope: "model")
        
    } else {
        print(">>>> no SummaryLayer >>>>>")
    }
    
    print("\n\n")

}
