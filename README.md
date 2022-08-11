<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/yukyeongleee/Wav2Lip-HQ">
    <!-- <img src="assets/templates/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

<h3 align="center">High Quality Wav2Lip</h3>

  <p align="center">
    Project description will be added
    <br />
    <a href="https://github.com/yukyeongleee/Wav2Lip-HQ"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/yukyeongleee/Wav2Lip-HQ">View Demo</a>
    ·
    <a href="https://github.com/Wav2Lip-HQ/issues">Report Bug</a>
    ·
    <a href="https://github.com/yukyeongleee/Wav2Lip-HQ/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#tdl">TDL</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
High-quality Wav2Lip, which can be trained on **arbitrary datasets**. 
As long as the training and inference scripts, the scripts for the [required preprocessings](https://github.com/Rudrabha/Wav2Lip#training-on-datasets-other-than-lrs2) are provided.

### Preprocessing
#### Audio
- Convert the audio sampling rate to **16000 Hz**. 
- Compute and save the mel-spectrogram for each audio. 

#### Video
- Convert the video frame rate to **25 fps**. 
- Extract and save raw frames(no face detection) from each video.
- Compute the offset between each audio and video pair by using the [pretrained SyncNet][repo-syncnet]. The offset values are needed for the **sync-correction** of the dataset. 
- **[Not provided]** Estimate the face bounding box. Crop and save the bounding box region for each frame. 
  - **Recommendation.** Use any high performance face detection tool **rather than s3fd**(the one used in [here]()). I used [InsightFace](https://github.com/deepinsight/insightface). 


### Changes from the [official implementation][repo-wav2lip]
#### Dataset
- Any datasets are available.

#### GPU usage
- Multi-GPU training is supported.
- To avoid bottleneck, mel-spectrograms are computed and saved as .npy files beforehand. (Previously, STFT is computed everytime when the `__getitem__` function is called)
 

#### Model Architecture
- The `FaceEncoder` of SyncNet takes **48 x 48 lip region image**, rather than 48 x 96 lower half image. (conditioned by the `tighter_box` option) 


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Installation

```sh
pip install -r requirements.txt
```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Preprocess

To begin with, the audio files are resampled with the sampling rate of 16000Hz. Also, STFT is applied to the resampled audio signals to obtain corresponding mel-spectrograms.
```sh
cd scripts/preprocess
python process_audio.py
```

Since the video files downloaded from YouTube have different frame rates(FPS), we should equalize this rate. The terminal command `ffmpeg` is used for frame rate conversion. The video length remains the same after conversion, so the audio doesn't have to be modified.

``` sh
python process_video.py
```

#### Sync-correction (using the official SyncNet)
The [official SyncNet implementation][repo-syncnet] and its pretrained checkpoint are used for sync-correction. All the dependencies should be installed before moving on to the next step. 
``` sh
git clone https://github.com/joonson/syncnet_python.git
cd syncnet_python 
```

Two python files(*get_offset.py* and *newSyncNetInstance.py*) in the *scripts/preprocess/sync-correction* directory need to be located in the syncnet_python directory. The shift value that minimizes syncnet loss is selected as offset. The offset value obtained for each video is recorded in the *output/offset.csv* file. If the input videos are not separated into frame images, adding `--separate_frames` option at the end of the line will help you.
``` sh
python get_offset.py # --separate_frames
```


### Train

```sh
git clone https://github.com/yukyeongleee/Wav2Lip-HQ.git
cd Wav2Lip-HQ
python scripts/train_syncnet.py {run_id} # SyncNet training
python scripts/train_wav2lip.py {run_id} # Wav2Lip training
```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- TDL -->
## TDL

- [x] Add dataset preprocessing scripts
- [x] Add sync-correction scripts

See the [open issues](https://github.com/yukyeongleee/Wav2Lip-HQ/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Contact us: ukok828@gmail.com

Project Link: [https://github.com/yukyeongleee/Wav2Lip-HQ](https://github.com/yukyeongleee/Wav2Lip-HQ)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Rudrabha/Wav2Lip][repo-wav2lip]: The official implementation
* [joonson/syncnet_python][repo-syncnet]: The official implementation
* [Innerverz-AI/CodeTemplate](https://github.com/Innerverz-AI/CodeTemplate)
* [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/yukyeongleee/Wav2Lip-HQ.svg?style=for-the-badge
[contributors-url]: https://github.com/yukyeongleee/Wav2Lip-HQ/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Iyukyeongleee/Wav2Lip-HQ.svg?style=for-the-badge
[forks-url]: https://github.com/yukyeongleee/Wav2Lip-HQ/network/members
[stars-shield]: https://img.shields.io/github/stars/yukyeongleee/Wav2Lip-HQ.svg?style=for-the-badge
[stars-url]: https://github.com/yukyeongleee/Wav2Lip-HQ/stargazers
[issues-shield]: https://img.shields.io/github/issues/yukyeongleee/Wav2Lip-HQ.svg?style=for-the-badge
[issues-url]: https://github.com/yukyeongleee/Wav2Lip-HQ/issues
[product-screenshot]: images/screenshot.png

[repo-wav2lip]: https://github.com/Rudrabha/Wav2Lip
[repo-syncnet]: https://github.com/joonson/syncnet_python