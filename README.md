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

### Changes from the [official implementation](https://github.com/Rudrabha/Wav2Lip)
- More efficient GPU usage
  - Multi-GPU supported.
  - To avoid bottleneck, mel-spectrograms are computed and saved as .npy files beforehand. (Previously, STFT is computed everytime when the `__getitem__` function is called)
- Available for arbitrary datasets
  - The codes for dataset preprocessing(fps and sync corrections) are provided. 
- Readability increased!

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

This is an example of how to list things you need to use the software and how to install them.

```sh
pip install -r requirements.txt
```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Preprocess

```sh
cd scripts/preprocess
```

To begin with, the audio files are resampled with the sampling rate of 16000Hz. Also, STFT is applied to the resampled audio signals to obtain corresponding mel-spectrograms.

``` sh
python process_audio.py
```

Since the video files downloaded from YouTube have different frame rates(FPS), we should equalize this rate. The terminal command `ffmpeg` is used for frame rate conversion. The video length remains the same after conversion, so the audio doesn't have to be modified.

``` sh
python process_video.py
```

To avoid the bottleneck at the data loading period, we crop the face region from each frame and save it. (No resizing, Not square) 
``` sh
python extract_face.py
```

**[For SyncNet Traning]** Training [SyncNet](https://github.com/joonson/syncnet_python) requires images of 224 x 224. The face images obtained by the previous step are padded to make a square and then resized.

``` sh
python resize_face.py
```

### Train

```sh
git clone https://github.com/Innerverz-AI/CodeTemplate.git
cd CodeTemplate 
python scripts/train.py {run_id}

# ex) python scripts/train.py first_try
```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- TDL -->
## TDL

- [x] Add dataset preprocessing code
- [ ] Add sync-correction code

See the [open issues](https://github.com/Innerverz-AI/CodeTemplate/issues) for a full list of proposed features (and known issues).

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

* [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip): The official implementation
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
