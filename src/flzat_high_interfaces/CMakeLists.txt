cmake_minimum_required(VERSION 3.8)
project(flzat_high_interfaces)

find_package(ament_cmake REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MemoryObject.msg"
  "msg/RoutedCommand.msg"
  "msg/SpeechAudio.msg"
  "srv/GetFrameNear.srv"
  "srv/QueryMemory.srv"
  "action/RunVlm.action"
  "action/SpeechToText.action"
  "action/TextToSpeech.action"
  DEPENDENCIES action_msgs builtin_interfaces geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
