// https://github.com/tensorflow/swift-apis/issues/25


import Foundation
import Python
import TensorFlow

public struct MyModel : Layer {
    public var conv1d: Conv1D<Float>
    public var dense1: Dense<Float>
    public var dropout: Dropout<Float>
    public var denseOut: Dense<Float>
    
    public init() {
        self.conv1d = Conv1D<Float>(filterShape: (2, 300, 100))

        self.dense1 = Dense<Float>(inputSize: 100,
                                   outputSize: 50,
                                   activation: relu)

        self.dropout = Dropout<Float>(probability: 0.02)

        self.denseOut = Dense<Float>(inputSize: 50, 
                                     outputSize: 2, 
                                     activation: sigmoid)
    }    

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let l1 = self.conv1d(input)
        let l2 = self.dense1(l1)
        let l3 = self.dropout(l2)
        let out = self.denseOut(l3)

        return out.squeezingShape()
    } 
}

extension Layer {
    mutating public func loadWeights(numpyFile: String) {
        let np = Python.import("numpy")
        let weights = np.load(numpyFile, allow_pickle: true)

        for (index, kp) in self.recursivelyAllWritableKeyPaths(to:  Tensor<Float>.self).enumerated() {
            self[keyPath: kp] = Tensor<Float>(numpy: weights[index])!
        }
    }

    public func saveWeights(numpyFile: String) {
        var weights: Array<PythonObject> = []

        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            weights.append(self[keyPath: kp].makeNumpyArray())
        }

        let np = Python.import("numpy")
        np.save(numpyFile, np.array(weights))
    }
}