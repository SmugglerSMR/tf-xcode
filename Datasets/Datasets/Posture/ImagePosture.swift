// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


import Foundation
import ModelSupport
import TensorFlow

/// Три варианта Imagenette, определяются по размеру исходного изображения.
public enum ImagePostureSize {
  case full
  case resized150
  case resized320

  var suffix: String {
    switch self {
    case .full: return ""
    case .resized150: return "-150"
    case .resized320: return "-320"
    }
  }
}

public struct ImagePosture<Entropy: RandomNumberGenerator> {
  /// Тип коллекции несборных партий.
  public typealias Batches = Slices<Sampling<[(file: URL, label: Int32)], ArraySlice<Int>>>
  /// Тип обучающих данных, представленных в виде последовательности эпох, которые представляют собой набор партий.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[(file: URL, label: Int32)], Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// Тип данных проверки, представленных в виде набора пакетов.
  public typealias Validation = LazyMapSequence<Slices<[(file: URL, label: Int32)]>, LabeledImage>
  /// Пакеты обучения.
  public let training: Training
  /// Пакеты проверки.
  public let validation: Validation

  /// Создает экземпляр с`batchSize`.
  ///
  /// - Parameters:
  ///   - batchSize: количество изображений, предоставляемых на пакет.
  ///   - entropy:   источник случайности, используемый для перемешивания порядка выборки. Это
  ///                будет храниться в `self`, поэтому, если он только псевдослучайный и имеет значение
  ///                семантика, последовательность эпох детерминирована и не зависит от других операций.
  ///   - device:    устройство, на котором будут также размещаться результирующие тензоры из этого набора данных
  ///                как то, где будут выполнены последние этапы любых расчетов конверсии.
  public init(batchSize: Int, entropy: Entropy, device: Device) {
    self.init(batchSize: batchSize, entropy: entropy, device: device, inputSize: ImagePostureSize.resized320, outputSize: 224)
  }

  /// Создает экземпляр с `batchSize` на` device`, используя `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - batchSize: количество изображений, предоставляемых на пакет.
  ///   - entropy:   источник случайности, используемый для перемешивания порядка выборки. Это
  ///                будет храниться в `self`, поэтому, если он только псевдослучайный и имеет значение
  ///                семантика, последовательность эпох детерминирована и не зависит от других операций.
  ///   - device:    устройство, на котором будут также размещаться результирующие тензоры из этого набора данных
  ///                как то, где будут выполнены последние этапы любых расчетов конверсии.
  ///   - inputSize: какой вариант размера изображения Imagenette использовать.
  ///   - outputSize: квадратная ширина и высота изображений, возвращаемых из этого набора данных.
  ///   - localStorageDirectory: где разместить загруженный и разархивированный набор данных.
  public init(
    batchSize: Int, entropy: Entropy, device: Device, inputSize: ImagePostureSize,
    outputSize: Int,
    localStorageDirectory: URL = DatasetUtilities.defaultDirectory.appendingPathComponent("ImagePosture", isDirectory: true)
  ) {
    do {
      //  считаем данные из каталога
      print("localStorageDirectory = \(localStorageDirectory)")
      let trainingSamples = try loadImagePostureTrainingDirectory(inputSize: inputSize, localStorageDirectory: localStorageDirectory, base: "imageposture")

      let mean = Tensor<Float>([0.485, 0.456, 0.406], on: device)
      let standardDeviation = Tensor<Float>([0.229, 0.224, 0.225], on: device)

      training = TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
        .lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledImage> in
          return batches.lazy.map {
            makeImagePostureBatch(
              samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
              device: device)
          }
        }

      let validationSamples = try loadImagePostureValidationDirectory(inputSize: inputSize, localStorageDirectory: localStorageDirectory, base: "imageposture")

      validation = validationSamples.inBatches(of: batchSize).lazy.map {
        makeImagePostureBatch(
          samples: $0, outputSize: outputSize, mean: mean, standardDeviation: standardDeviation,
          device: device)
      }
    } catch {
      fatalError("Could not load ImagePosture dataset: \(error)")
    }
  }
}

extension ImagePosture: ImageClassificationData where Entropy == SystemRandomNumberGenerator {
  /// Создает экземпляр с помощью `batchSize`, используя SystemRandomNumberGenerator.
  public init(batchSize: Int, on device: Device = Device.default) {
    self.init(batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device)
  }

  /// Создает экземпляр с `batchSize`,` inputSize` и `outputSize`, используя SystemRandomNumberGenerator.
  public init(
    batchSize: Int, inputSize: ImagePostureSize, outputSize: Int, on device: Device = Device.default
  ) {
    self.init(
      batchSize: batchSize, entropy: SystemRandomNumberGenerator(), device: device,
      inputSize: inputSize, outputSize: outputSize)
  }
}

func downloadImagePostureIfNotPresent(to directory: URL, size: ImagePostureSize, base: String) {
  let downloadPath = directory.appendingPathComponent("\(base)\(size.suffix)").path
  let directoryExists = FileManager.default.fileExists(atPath: downloadPath)
  let contentsOfDir = try? FileManager.default.contentsOfDirectory(atPath: downloadPath)
  let directoryEmpty = (contentsOfDir == nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return }

  //let location = URL(string: "https://s3.amazonaws.com/fast-ai-imageclas/\(base)\(size.suffix).tgz")!
  let location = URL(string: "https://znanija.info/assets/files/imageposture.tgz")!
  let _ = DatasetUtilities.downloadResource(
    filename: "\(base)\(size.suffix)", fileExtension: "tgz",
    remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)
}

func exploreImagePostureDirectory(named name: String, in directory: URL, inputSize: ImagePostureSize, base: String) throws -> [URL] {
  downloadImagePostureIfNotPresent(to: directory, size: inputSize, base: base)
  let path = directory.appendingPathComponent("\(base)\(inputSize.suffix)/\(name)")
  let dirContents = try FileManager.default.contentsOfDirectory(at: path, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])

  var urls: [URL] = []
  for directoryURL in dirContents {
    let subdirContents = try FileManager.default.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles])
    urls += subdirContents
  }
  return urls
}

//func parentLabel(url: URL) -> String {
//  return url.deletingLastPathComponent().lastPathComponent
//}
//
//func createLabelDict(urls: [URL]) -> [String: Int] {
//  let allLabels = urls.map(parentLabel)
//  let labels = Array(Set(allLabels)).sorted()
//  return Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, $0.offset) })
//}

func loadImagePostureDirectory(named name: String, in directory: URL, inputSize: ImagePostureSize, base: String,
                               labelDict: [String: Int]? = nil) throws -> [(file: URL, label: Int32)] {
    
  let urls = try exploreImagePostureDirectory(named: name, in: directory, inputSize: inputSize, base: base)
  let unwrappedLabelDict = labelDict ?? createLabelDict(urls: urls)
  return urls.lazy.map { (url: URL) -> (file: URL, label: Int32) in
    (file: url, label: Int32(unwrappedLabelDict[parentLabel(url: url)]!))
  }
}

func loadImagePostureTrainingDirectory(inputSize: ImagePostureSize, localStorageDirectory: URL, base: String,
                                       labelDict: [String: Int]? = nil) throws  -> [(file: URL, label: Int32)]
{
  return try loadImagePostureDirectory(
    named: "train", in: localStorageDirectory, inputSize: inputSize, base: base,
    labelDict: labelDict)
}

func loadImagePostureValidationDirectory(inputSize: ImagePostureSize, localStorageDirectory: URL, base: String,
                                         labelDict: [String: Int]? = nil) throws  -> [(file: URL, label: Int32)]
{
  return try loadImagePostureDirectory(
    named: "val", in: localStorageDirectory, inputSize: inputSize, base: base, labelDict: labelDict)
}

func makeImagePostureBatch<BatchSamples: Collection>(
  samples: BatchSamples, outputSize: Int, mean: Tensor<Float>?, standardDeviation: Tensor<Float>?,
  device: Device
) -> LabeledImage where BatchSamples.Element == (file: URL, label: Int32) {
  let images = samples.map(\.file).map { url -> Tensor<Float> in
    Image(jpeg: url).resized(to: (outputSize, outputSize)).tensor
  }

  var imageTensor = Tensor(stacking: images)
  imageTensor = Tensor(copying: imageTensor, to: device)
  imageTensor /= 255.0

  if let mean = mean, let standardDeviation = standardDeviation {
    imageTensor = (imageTensor - mean) / standardDeviation
  }

  let labels = Tensor<Int32>(samples.map(\.label), on: device)
  return LabeledImage(data: imageTensor, label: labels)
}
