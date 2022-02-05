//
//  ImageClassificationModelsWriter.swift
//  ImageClassificationModels
//
//  Created by Рамиль Садыков on 19.06.2020.
//  Copyright © 2020 Sadukow. All rights reserved.
//

import Foundation

import ModelSupport
import TensorFlow

import Python

protocol ExportableLayer {
    var nameMappings: [String: String] { get }
    func save_weight(scope: String, tensors: inout [String: Tensor<Float>])
}







extension GlobalAvgPool2D: ExportableLayer {
    var nameMappings: [String: String] { [:] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        print("--GlobalAvgPool2D:save_weight-- \(scope)")
    }
}

extension ZeroPadding2D: ExportableLayer {
    var nameMappings: [String: String] { [:] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        print("--ZeroPadding2D:save_weight-- \(scope)")
    }
}

extension BatchNorm: ExportableLayer {
    var nameMappings: [String: String] { ["scale": "w", "offset": "b"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--BatchNorm:save_weight-- \(scope)")
        if let tensor = self.offset as? Tensor<Float> {
            let path = scope + "/b"
            //print("\(String(describing: type(of: self))) .offset - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.scale as? Tensor<Float> {
            let path = scope + "/w"
            //print("\(String(describing: type(of: self))) .scale - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.runningMean.value as? Tensor<Float> {
            let path = scope + "/m"
            //print("\(String(describing: type(of: self))) .runningMean - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.runningVariance.value as? Tensor<Float> {
            let path = scope + "/v"
            //print("\(String(describing: type(of: self))) .runningVariance - >>>>>  \(path)")
            tensors[path] = tensor
        }
    }
}

extension DepthwiseConv2D: ExportableLayer {
    var nameMappings: [String: String] { ["filter": "w", "bias": "b"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--DepthwiseConv2D:save_weight-- \(scope)")
        if let tensor = self.filter as? Tensor<Float> {
            let path = scope + "/w"
            //print("\(String(describing: type(of: self))) .filter - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.bias as? Tensor<Float> {
            let path = scope + "/b"
            //print("\(String(describing: type(of: self))) .bias - >>>>>  \(path)")
            tensors[path] = tensor
        }
    }
}

extension Conv2D: ExportableLayer {
    var nameMappings: [String: String] { ["filter": "w", "bias": "b"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--Conv2D:save_weight-- \(scope)")
        if let tensor = self.filter as? Tensor<Float> {
            let path = scope + "/w"
            //print("\(String(describing: type(of: self))) .filter - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.bias as? Tensor<Float> {
            let path = scope + "/b"
            //print("\(String(describing: type(of: self))) .bias - >>>>>  \(path)")
            tensors[path] = tensor
        }
    }
}

extension LayerNorm: ExportableLayer {
    var nameMappings: [String: String] { ["offset": "b", "scale": "w"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--LayerNorm:save_weight-- \(scope)")
        if let tensor = self.offset as? Tensor<Float> {
            let path = scope + "/b"
            //print("\(String(describing: type(of: self))) .offset - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.scale as? Tensor<Float> {
            let path = scope + "/w"
            //print("\(String(describing: type(of: self))) .scale - >>>>>  \(path)")
            tensors[path] = tensor
        }
    }
}

extension Dense: ExportableLayer {
    var nameMappings: [String: String] { ["weight": "w", "bias": "b"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        //print("--Dense:save_weight-- \(scope)")
        if let tensor = self.weight as? Tensor<Float> {
            let path = scope + "/w"
            //print("\(String(describing: type(of: self))) .weight - >>>>>  \(path)")
            tensors[path] = tensor
        }
        if let tensor = self.bias as? Tensor<Float> {
            let path = scope + "/b"
            //print("\(String(describing: type(of: self))) .bias - >>>>>  \(path)")
            tensors[path] = tensor
        }
    }
}

extension Array: ExportableLayer {
    var nameMappings: [String: String] { ["h": "h"] }
    public func save_weight(scope: String, tensors: inout [String: Tensor<Float>]) {
        print("--Array::save_weight-- \(scope)")
    }
}


public func recursivelyObtainTensors(
    _ obj: Any, scope: String? = nil, tensors: inout [String: Tensor<Float>], separator: String
) {
    let m = Mirror(reflecting: obj)
    let nameMappings: [String: String]
    
    //print("\n recursivelyObtainTensors:: scope= \(scope!) separator= \(separator) ")
    
    //print(String(describing: obj))
    //print(String(describing: type(of: obj)))

    if let exportableLayer = obj as? ExportableLayer {
        nameMappings = exportableLayer.nameMappings
    } else {
        nameMappings = [:]
    }
    //print(nameMappings)

    var repeatedLabels: [String: Int] = [:]
    func suffix(for label: String) -> String {
        if let currentSuffix = repeatedLabels[label] {
            repeatedLabels[label] = currentSuffix + 1
            return "\(currentSuffix + 1)"
        } else {
            repeatedLabels[label] = 0
            return "0"
        }
    }

    let hasSuffix = (m.children.first?.label == nil)
    

    var path = scope
    for child in m.children {
        let label = child.label ?? "h"

        if let remappedLabel = nameMappings[label] {
            //print("remappedLabel = \(remappedLabel)")
            let labelSuffix = hasSuffix ? suffix(for: remappedLabel) : ""
            //print("labelSuffix = \(labelSuffix)")
            let conditionalSeparator = remappedLabel == "" ? "" : separator

            path = (scope != nil ? scope! + conditionalSeparator : "") + remappedLabel + labelSuffix
            if let tensor = child.value as? Tensor<Float> {
                //String(describing: obj)
                print("\(String(describing: type(of: obj))) - \(label) >>>>>  \(path!)")
                tensors[path!] = tensor
            }
        }
        recursivelyObtainTensors(child.value, scope: path, tensors: &tensors, separator: separator)
    }
    
}

public func ImageClassificationWriteCheckpoint(model: Any, to location: URL, name: String) throws {
    var tensors = [String: Tensor<Float>]()
//    recursivelyObtainTensors(model, scope: "model", tensors: &tensors, separator: "/")
    if let ExportableLayer = model as? ExportableLayer {
       ExportableLayer.save_weight(scope: "model", tensors: &tensors)
    } else {
       print(">>>> no ExportableLayer >>>>>")
    }
    

    let writer = CheckpointWriter(tensors: tensors)
    try writer.write(to: location, name: name)
}

//extension Layer {
//    public func saveWeights(scope: String, tensors: inout [String: Tensor<Float>]) {
//        print("\(String(describing: type(of: self))) - >>>>>  \(scope)")
//        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
//            print(kp)
//            //weights.append(self[keyPath: kp].makeNumpyArray())
//        }
//    }
//}
