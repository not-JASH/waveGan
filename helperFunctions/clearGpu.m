function resetGpus
    for i = 1:gpuDeviceCount
        gpuDevice(i);
        clc;
    end
end