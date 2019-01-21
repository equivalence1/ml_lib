#pragma once

enum class ComputeDeviceType {
    Cpu,
    Gpu
};

class ComputeDevice {
public:
    ComputeDevice(ComputeDeviceType type)
        : deviceType_(type) {

    }

    ComputeDeviceType deviceType() const {
        return deviceType_;
    }

    bool operator==(const ComputeDevice& rhs) const {
        return deviceType_ == rhs.deviceType_;
    }
    bool operator!=(const ComputeDevice& rhs) const {
        return !(rhs == *this);
    }
private:
    ComputeDeviceType deviceType_;
};

const ComputeDevice& CurrentDevice();

void SetDevice(const ComputeDevice& device);

class ComputeDeviceGuard {

public:

    ComputeDeviceGuard(const ComputeDevice& toUse)
        : prev_(CurrentDevice())
          , current_(toUse) {
        SetDevice(current_);

    }

    ~ComputeDeviceGuard() {
        SetDevice(prev_);
    }

    operator bool() const {
        return true;
    }
private:
    ComputeDevice prev_;
    ComputeDevice current_;
};



/*
 * on_device(device) {
 *   ...
 * }
 */
#define on_device(device)                                                   \
    if (auto UNIQUE_ID(var) = ComputeDeviceGuard(device)) {                 \
        goto CONCAT(THIS_IS_DEVICE_GUARD, __LINE__);                        \
    } else                                                                  \
        CONCAT(THIS_IS_DEVICE_GUARD, __LINE__)                              \
            :
