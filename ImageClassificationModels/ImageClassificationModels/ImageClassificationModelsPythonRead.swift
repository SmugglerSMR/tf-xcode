//
//  ImageClassificationModelsPythonRead.swift
//  ImageClassificationModels
//
//  Created by Рамиль Садыков on 19.06.2020.
//  Copyright © 2020 Sadukow. All rights reserved.
//

import Foundation

import ModelSupport
import TensorFlow

public typealias ImportMap = [String: (String, [Int]?)]

public let layer_with_weights = "layer_with_weights-{layer}/{value}/.ATTRIBUTES/VARIABLE_VALUE"

protocol ImportablePythonLayer {
    mutating func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int)
}

func getLayerName(key: String, number: Int) -> String {
    let layer = String(number)
    let label = layer_with_weights.replacingOccurrences(of: "{layer}", with: layer)
                                  .replacingOccurrences(of: "{value}", with: key)
    return label
}


extension DepthwiseConv2D: ImportablePythonLayer {
    mutating func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("DepthwiseConv2D.load_python_weight    \(scope) <<<< layer_with_weights-\(number)")
        
        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "depthwise_kernel", number: number)],
                     tensor: &filter )
        
//        load_weight( reader: reader,
//                     labels: [scope+getLayerName(key: "bias", number: number)],
//                     tensor: &bias )

        number = number + 1
    }
}


extension Conv2D: ImportablePythonLayer {
    mutating func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("Conv2D.load_python_weight     <<<< \(scope)layer_with_weights-\(number)")
        
        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "kernel", number: number)],
                     tensor: &filter )

//        load_weight( reader: reader,
//                     labels: [scope+getLayerName(key: "bias", number: number)],
//                     tensor: &bias )

        number = number + 1
    }
}


extension Dense: ImportablePythonLayer {
    mutating func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("Dense.load_python_weight     <<<< \(scope)layer_with_weights-\(number)")

        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "kernel", number: number)],
                     tensor: &weight )
        
        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "bias", number: number)],
                     tensor: &bias )

        number = number + 1
    }
}

extension LayerNorm: ImportablePythonLayer {
    mutating func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("LayerNorm.load_python_weight    \(scope) <<<< layer_with_weights-\(number)")

        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "kernel", number: number)],
                     tensor: &offset )
        
        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "bias", number: number)],
                     tensor: &scale )

        number = number + 1
    }
}


extension BatchNorm: ImportablePythonLayer {
    mutating func load_python_weight(reader: CheckpointReader, scope: String, number: inout Int) {
        //print("BatchNorm.load_python_weight    \(scope) <<<< layer_with_weights-\(number)")
        
        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "gamma", number: number)],
                     tensor: &scale )

        load_weight( reader: reader,
                     labels: [scope+getLayerName(key: "beta", number: number)],
                     tensor: &offset )

        load_weight_1( reader: reader,
                       labels: [scope+getLayerName(key: "moving_mean", number: number)],
                       tensor: &runningMean.value )

        load_weight_1( reader: reader,
                       labels: [scope+getLayerName(key: "moving_variance", number: number)],
                       tensor: &runningVariance.value )

        number = number + 1
    }
}


