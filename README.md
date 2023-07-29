# obs-cleanstream
**🚧 EXPERIMENTAL 🚧**

CleanStream is an OBS plugin that cleans live audio streams from unwanted words and utterances using AI.

**🚧 EXPERIMENTAL 🚧**

<div align="center">

[![GitHub](https://img.shields.io/github/license/royshil/obs-cleanstream)](https://github.com/royshil/obs-cleanstream/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/royshil/obs-cleanstream/push.yaml)](https://github.com/royshil/obs-cleanstream/actions/workflows/push.yaml)
[![Total downloads](https://img.shields.io/github/downloads/royshil/obs-cleanstream/total)](https://github.com/royshil/obs-cleanstream/releases)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/royshil/obs-cleanstream)](https://github.com/royshil/obs-cleanstream/releases)

</div>

Check out our other plugin: [OBS Background Removal](https://github.com/royshil/obs-backgroundremoval)

## Download
Check out the [latest releases](https://github.com/royshil/obs-cleanstream/releases) for downloads and install instructions.

## Code Walkthrough
This video walkthrough (YouTube) will explain various parts of the code of you're looking to learn from what I've discovered.

<a href="https://youtu.be/HdSI3sUKwsY" target="_blank">
  <img width="480" src="https://img.youtube.com/vi/HdSI3sUKwsY/maxresdefault.jpg" />
</a>

## Requirements
- OBS version 28+ ([download](https://obsproject.com/download))

We do not support older versions of OBS since the plugin is using newer APIs.

## Introduction
CleanStream is an OBS plugin that cleans live audio streams from unwanted words and utterances, such as "uh"s and "um"s, and other words that you can configure, like profanity.

See our [resource on the OBS Forums](https://obsproject.com/forum/resources/cleanstream-remove-uhs-ums-profanity-in-your-live-stream-or-recording-with-ai.1732/) for additional information.

It is using a neural network ([OpenAI Whisper](https://github.com/openai/whisper)) to predict in real time the speech and remove the unwanted words.

It's using the [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) project from [ggerganov](https://github.com/ggerganov) to run the Whisper network in a very efficient way.

### 🚧🚧🚧 **This plugin is still experimental and is not ready for live production use.** 🚧🚧🚧

But it is working and you can try it out. *Please report any issues you find.* 🙏 ([submit an issue](https://github.com/royshil/obs-cleanstream/issues))

We're working on improving the plugin and adding more features. If you have any ideas or suggestions, please open an issue.

GPU support is coming soon. Whisper.cpp is using GGML which should have GPU support for major platforms. We will bring it to the plugin when it's ready.

## Building

The plugin was built and tested on Mac OSX, Windows and Ubuntu Linux. Help is appreciated in building on other OSs and packages.

The building pipelines in CI take care of the heavy lifting. Use them in order to build the plugin locally.

Start by cloning this repo to a directory of your choice.

### Mac OSX

Using the CI pipeline scripts, locally you would just call the zsh script.

```sh
$ ./.github/scripts/build-macos.zsh -c Release -t macos-x86_64
```

#### Install
The above script should succeed and the plugin files will reside in the `./release` folder off of the root. Copy the files to the OBS directory e.g. `/Users/you/Library/Application Support/obs-studio/obs-plugins`.

To get `.pkg` installer file, run
```sh
$ ./.github/scripts/package-macos.zsh -c Release -t macos-x86_64
```
(Note that maybe the outputs in the e.g. `build_x86_64` will be in the `Release` folder and not the `install` folder like `pakage-macos.zsh` expects, so you will need to rename the folder from `build_x86_64/Release` to `build_x86_64/install`)

### Linux (Ubuntu-ish)

Use the CI scripts again
```sh
$ ./.github/scripts/build-linux.sh
```

### Windows

Use the CI scripts again, for example:

```powershell
> .github/scripts/Build-Windows.ps1 -Target x64 -CMakeGenerator "Visual Studio 17 2022"
```

The build should exist in the `./release` folder off the root. You can manually install the files in the OBS directory, e.g. `C:\Program Files\obs-studio\obs-plugins`.

