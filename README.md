
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">

<h3 align="center">RL Implementations</h3>

  <p align="center">
    Reinforcement Learning Algorithm implementations in Jax (Equinox)
  </p>
</div>


## About The Project
A Repository containing (mostly) single-file reinforcement learning implementations in Jax. The repository is heavily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [PureJaxRL](https://github.com/luchris429/purejaxr). Like those libraries, the implementations are not modular by design; Instead, the code focusses on clarity allowing you to alter it to your own needs. In contrast to both mentioned libraries, this repository implements its algorithms in [Equinox](https://github.com/patrick-kidger/equinox).

This repository is heavily a work in progress for personal use. Use it as you see fit. Suggestions and/or improvements are of course welcome.

## Useage

### Prerequisites

Install dependencies using the `requirements.txt` file:

  ```sh
  pip install -r requirements.txt
  ```

For CUDA enabled devices you may want to uncomment part of the requirements.txt file to use JAX on your accelerators. For more information, visit the [JAX documentation](https://github.com/google/jax#installation).


### Structure

Most algorithm code lives in a single file under the `algorithms` folder; However, the networks themselves are seperated in `util/networks`.

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ponseko/rl-implementations.svg?style=for-the-badge
[contributors-url]: https://github.com/ponseko/rl-implementations/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ponseko/rl-implementations.svg?style=for-the-badge
[forks-url]: https://github.com/ponseko/rl-implementations/network/members
[stars-shield]: https://img.shields.io/github/stars/ponseko/rl-implementations.svg?style=for-the-badge
[stars-url]: https://github.com/ponseko/rl-implementations/stargazers
[issues-shield]: https://img.shields.io/github/issues/ponseko/rl-implementations.svg?style=for-the-badge
[issues-url]: https://github.com/ponseko/rl-implementations/issues
[license-shield]: https://img.shields.io/github/license/ponseko/rl-implementations.svg?style=for-the-badge
[license-url]: https://github.com/ponseko/rl-implementations/blob/main/LICENSE
