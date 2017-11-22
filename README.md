# Automatic ICD-9 coding via deep transfer learning

## Environment Variables setup:  

LD_LIBRARY_PATH		/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH  

CUDA_HOME 	/usr/local/cuda
CUDA_DEVICE_ORDER 	PCI_BUS_ID
CUDA_VISIABLE_DEVICES 	0,1,2,3  

TF_CPP_MIN_LOG_LEVEL	1  # 显示信息的层级（显示全部，只显示 warning 和 Error，只显示Error）  

**equate to:**
import os
os.environ["CUDA_VISIABLE_DEVICES"]="3"