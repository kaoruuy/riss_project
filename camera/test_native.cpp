#include <sl/Camera.hpp>

#include <iostream>

int main() {
    const auto devices = sl::Camera::getDeviceList();
    std::cout << "ZED SDK: " << sl::Camera::getSDKVersion() << '\n';
    std::cout << "ZED devices reported by SDK: " << devices.size() << '\n';

    for (const auto& device : devices) {
        std::cout << "  id=" << device.id << ", serial=" << device.serial_number
                  << ", model=" << sl::toString(device.camera_model)
                  << ", state=" << sl::toString(device.camera_state) << '\n';
    }

    sl::Camera zed;
    sl::InitParameters parameters;
    parameters.camera_resolution = sl::RESOLUTION::HD720;
    parameters.camera_fps = 30;
    parameters.depth_mode = sl::DEPTH_MODE::NONE;
    parameters.sdk_verbose = 1;

    std::cout << "Opening camera at HD720/30 with depth disabled..." << std::endl;
    const auto result = zed.open(parameters);
    std::cout << "Open result: " << sl::toString(result) << '\n';
    zed.close();

    return result == sl::ERROR_CODE::SUCCESS ? 0 : 1;
}
