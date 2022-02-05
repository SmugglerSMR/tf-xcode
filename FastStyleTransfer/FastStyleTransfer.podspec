#
#  Be sure to run `pod spec lint FastStyleTransfer.podspec' to ensure this is a
#  valid spec and to remove all comments including this before submitting the spec.

Pod::Spec.new do |spec|

  # ―――  Spec Metadata  ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.name         = "FastStyleTransfer"
  spec.version      = "0.0.1"
  spec.summary      = "A short description of FastStyleTransfer."
  spec.description  = "FastStyleTransfer for TensorFlow"
  spec.homepage     = "http://sadukow.com/FastStyleTransfer"


  # ―――  Spec License  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.license      = "MIT"


  # ――― Author Metadata  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.author             = { "Sadukow" => "sadukow@gmail.com" }

  # ――― Platform Specifics ――――――――――――――――――――――――――――――――――――――――――――――――――――――― #
  
  # spec.platform     = :ios
  # spec.osx.deployment_target = "10.7"
  spec.platform     = :osx, '10.15'
  spec.osx.deployment_target = "10.15"


  # ――― Source Location ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.source       = { :path => '.'  }


  # ――― Source Code ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.source_files  = "FastStyleTransfer", "FastStyleTransfer/**/*.{h,m,swift}", "Classes", "Classes/**/*.{h,m}"
  spec.exclude_files = "Classes/Exclude"
  #spec.public_header_files      = "FastStyleTransfer.framework/Headers/*.h", "FastStyleTransfer.framework/SubFrameworks/**/Headers/*.h", 


  # ――― Resources ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  #spec.resources        = "ModelSupport/**/*"
  # spec.resource  = "icon.png"
  # spec.resources = "Resources/*.png"

  # spec.preserve_paths = "FilesToSave", "MoreFilesToSave"


  # ――― Project Linking ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  #spec.framework  = "ModelSupport"
  # spec.frameworks = "ModelSupport", "SwiftProtobuf"

  # spec.library   = "iconv"
  # spec.libraries = "iconv", "xml2"


  # ――― Project Settings ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  spec.requires_arc = true
  #spec.xcconfig = { "HEADER_SEARCH_PATHS" => "$(SDKROOT)/usr/include/libxml2" }
  #spec.dependency 	'SwiftProtobuf', '~> 1.0'
  spec.dependency 	'ModelSupport'

end
