![image](https://github.com/liutao14/Realtime_Video_Stabilization/assets/110754123/54970b7f-6e69-4e6f-af40-52a149e5837e)![image](https://github.com/liutao14/Realtime_Video_Stabilization/assets/110754123/7ed93920-cc3b-4c71-917d-75f7181b05c4)# Realtime_Video_Stabilization
This repository contains the source code implementation for our paper titled "Real-Time Video Stabilization Algorithm Based on SuperPoint." The paper presents an efficient algorithm for stabilizing shaky videos by leveraging SuperPoint feature points.

Paper Link: https://webvpn.njust.edu.cn/https/77726476706e69737468656265737421f9f244993f20645f6c0dc7a59d50267b1ab4a9/document/10360180/

Important Notes
Please note that this repository is not a direct replica of the version described in the paper. We have made enhancements and included new features such as alike and zippypoint for extracting image features. Additionally, we have incorporated dense optical flow, pwcnet, and liteflownet for tracking the movement of feature points. Furthermore, LSTM is utilized to smoothen the camera path. However, please be advised that this repository is still under development, and some functionalities may not be fully complete.

Usage Scope
This code is solely intended for academic research purposes and is strictly prohibited from being used for any commercial or non-academic endeavors.

Citation Instructions
If you find this code useful for your research, please cite our paper as follows:

@article{Liu_Wan_Bai_Kong_Tang_Wang_2024,  
  title={Real-Time Video Stabilization Algorithm Based on SuperPoint},  
  volume={73},  
  DOI={10.1109/tim.2023.3342849},  
  journal={IEEE Transactions on Instrumentation and Measurement},  
  author={Liu, Tao and Wan, Gang and Bai, Hongyang and Kong, Xiaofang and Tang, Bo and Wang, Fangyi},  
  year={2024},  
  month={Jan},  
  pages={1–13},  
  language={English}  
}
Running the Code
To run the code in this repository, please follow the steps below:

Clone this repository to your local environment.

Install the required dependencies by referring to the requirements.txt file.

Modify the video paths and uncomment the relevant code in stab.py to match your local setup and requirements.

Execute the stab.py script.

bash
python stab.py
Please ensure that all dependencies and paths are properly configured before running the code.

***************Please organize the directory structure according to the given file ：directory.jpg***********************


Contributions and Feedback
If you encounter any issues while using this repository or wish to contribute improvements, please feel free to contact us via email or through GitHub issues. We welcome any feedback and suggestions.

Copyright Notice
All content in this repository is protected by copyright laws. Unauthorized duplication, distribution, or commercial use is strictly prohibited. By using this repository, you agree to abide by the terms stated above.
