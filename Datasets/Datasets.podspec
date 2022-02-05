#
#  Be sure to run `pod spec lint Datasets.podspec' to ensure this is a
#  valid spec and to remove all comments including this before submitting the spec.
#
#  To learn more about Podspec attributes see https://guides.cocoapods.org/syntax/podspec.html
#  To see working Podspecs in the CocoaPods repo see https://github.com/CocoaPods/Specs/
#

Pod::Spec.new do |spec|

  # ―――  Spec Metadata  ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.name         = "Datasets"
  spec.version      = "0.0.1"
  spec.summary      = "A short description of Datasets."
  spec.description  = "Datasets - framework for TensorFlow (swift-models)"
  spec.homepage     = "http://sadukow.com/Datasets"


  # ―――  Spec License  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.license      = "MIT"


  # ――― Author Metadata  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.author             = { "Sadukow" => "sadukow@gmail.com" }

  # ――― Platform Specifics ――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.platform     = :osx, '10.15'
  spec.osx.deployment_target = "10.15"


  # ――― Source Location ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.source       = { :path => '.'  }


  # ――― Source Code ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.source_files  = "Datasets", "Datasets/**/*.{h,m,swift}", "Classes", "Classes/**/*.{h,m}"
  spec.exclude_files = "Classes/Exclude"


  # ――― Resources ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #


  # ――― Project Linking ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #


  # ――― Project Settings ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.requires_arc = true
  spec.dependency 	'ModelSupport'
  spec.dependency 	'Batcher'


end
