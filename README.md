# obs-cleanstream

CleanStream is an OBS plugin that cleans live audio streams from unwanted words and utterances using real-time local AI.

<div align="center">

[![GitHub](https://img.shields.io/github/license/occ-ai/obs-cleanstream)](https://github.com/occ-ai/obs-cleanstream/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/occ-ai/obs-cleanstream/push.yaml)](https://github.com/occ-ai/obs-cleanstream/actions/workflows/push.yaml)
[![Total downloads](https://img.shields.io/github/downloads/occ-ai/obs-cleanstream/total)](https://github.com/occ-ai/obs-cleanstream/releases)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/occ-ai/obs-cleanstream)](https://github.com/occ-ai/obs-cleanstream/releases)
[![Discord](https://img.shields.io/discord/1200229425141252116)](https://discord.gg/KbjGU2vvUz)

</div>

Check out our other plugins:
- [Background Removal](https://github.com/occ-ai/obs-backgroundremoval) remove background (virtual green screen) from video
- [Detect](https://github.com/occ-ai/obs-detect) will detect and track >80 types of objects in any OBS source
- [URL/API Source](https://github.com/occ-ai/obs-urlsource) fetch API data and display it on screen as a video source
- [LocalVocal](https://github.com/occ-ai/obs-localvocal) speech AI assistant plugin for real-time transcription (captions), translation and more language functions
- [PolyGlot](https://github.com/occ-ai/obs-polyglot) a realtime local translation service based on AI.

## Usage

- Add the plugin to any audio-generating source
- Adjust the settings

<div align="center">

<a href="https://youtu.be/vgcK3K654FU"><img src="https://github.com/occ-ai/obs-cleanstream/assets/441170/59f0038a-4e0e-47ed-afb8-110acd463146" width="40%" /></a>

</div>

## Download
Check out the [latest releases](https://github.com/occ-ai/obs-cleanstream/releases) for downloads and install instructions.

## Code Walkthrough
This video walkthrough (YouTube) will explain various parts of the code of you're looking to learn from what I've discovered.

<div align="center">
    <a href="https://youtu.be/HdSI3sUKwsY" target="_blank">
        <img width="480" src="https://img.youtube.com/vi/HdSI3sUKwsY/maxresdefault.jpg" />
    </a>
</div>

## Requirements
- OBS version 30+ for plugin versions 0.0.4+
- OBS version 29 for plugin versions 0.0.2+
- OBS version 28 for plugin versions 0.0.1

We do not support older versions of OBS since the plugin is using newer APIs.

## Introduction
CleanStream is an OBS plugin that cleans live audio streams from unwanted words and utterances, such as "uh"s and "um"s, and other words that you can configure, like profanity.

See our [resource on the OBS Forums](https://obsproject.com/forum/resources/cleanstream-remove-uhs-ums-profanity-in-your-live-stream-or-recording-with-ai.1732/) for additional information.

It is using a neural network ([OpenAI Whisper](https://github.com/openai/whisper)) to predict in real time the speech and remove the unwanted words.

It's using the [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) project from [ggerganov](https://github.com/ggerganov) to run the Whisper network in a very efficient way.

But it is working and you can try it out. *Please report any issues you find.* 🙏 ([submit an issue](https://github.com/occ-ai/obs-cleanstream/issues) or meet us on https://discord.gg/KbjGU2vvUz)

We're working on improving the plugin and adding more features. If you have any ideas or suggestions, please open an issue.

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
> .github/scripts/Build-Windows.ps1 -Configuration Release
```

The build should exist in the `./release` folder off the root. You can manually install the files in the OBS directory.

```powershell
> Copy-Item -Recurse -Force "release\Release\*" -Destination "C:\Program Files\obs-studio\"
```

#### Building with CUDA support on Windows

CleanStream will now build with CUDA support automatically through a prebuilt binary of Whisper.cpp from https://github.com/occ-ai/occ-ai-dep-whispercpp. The CMake scripts will download all necessary files.

To build with cuda add `CPU_OR_CUDA` as an environment variable (with `cpu`, `12.2.0` or `11.8.0`) and build regularly

```powershell
> $env:CPU_OR_CUDA="12.2.0"
> .github/scripts/Build-Windows.ps1 -Configuration Release
```
