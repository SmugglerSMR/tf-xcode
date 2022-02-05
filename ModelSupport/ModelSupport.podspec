#
#  Be sure to run `pod spec lint ModelSupport.podspec' to ensure this is a
#  valid spec and to remove all comments including this before submitting the spec.
#
#  To learn more about Podspec attributes see https://guides.cocoapods.org/syntax/podspec.html
#  To see working Podspecs in the CocoaPods repo see https://github.com/CocoaPods/Specs/
#

Pod::Spec.new do |spec|

  spec.name         = "ModelSupport"
  spec.version      = "0.0.1"
  spec.summary      = "A short description of ModelSupport."
  spec.description  = "frameworks ModelSupport."
  spec.homepage     = "http://sadukow.com/ModelSupport"
  spec.license      = { :type => 'MIT' }
  spec.author             = { "Sadukow" => "sadukow@gmail.com" }
  spec.platform     = :osx, '10.15'
  spec.osx.deployment_target = "10.15"
  spec.source       = { :path => '.'  }
  spec.source_files     = "ModelSupport", "ModelSupport/**/*.{h,m,swift}", 'Classes/*'
  #spec.exclude_files    = "Classes/Exclude"
  #spec.public_header_files      = "ModelSupport.framework/Headers/*.h", "ModelSupport.framework/SubFrameworks/**/Headers/*.h"
  #spec.resources        = "ModelSupport/**/*"
  

  spec.dependency 	'SwiftProtobuf', '~> 1.0'
  spec.dependency 	'STBImage'

  #spec.xcconfig = {'ENABLE_BITCODE' => 'NO'}   #It is not recommended using s.xcconfig as it changes the users configuration. Use pod_target_xcconfig instead.


  #spec.osx.deployment_target   = '10.15'
  #spec.osx.vendored_frameworks = 'ModelSupport.framework'



  #spec.public_header_files      = "SwiftProtobuf.framework/Headers/*.h", "ModelSupport.framework/SubFrameworks/**/Headers/*.h"
  #spec.osx.deployment_target   = '9.0'
  #spec.osx.vendored_frameworks = 'ModelSupport.framework'

  #spec.library = 'SwiftProtobuf'
  #spec.xcconfig = { 'LIBRARY_SEARCH_PATHS' => "$(SRCROOT)/Pods/**" }
 
  #spec.framework        = 'SystemConfiguration'
  
  #spec.osx.vendored_frameworks = 'Frameworks/SwiftProtobuf.framework'
  #spec.vendored_frameworks  = 'ModelSupport.framework', 'SwiftProtobuf.framework'
  #spec.requires_arc     = true    
  

end
