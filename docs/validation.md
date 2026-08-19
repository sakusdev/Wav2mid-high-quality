# Build validation

This branch exists to run the production Vite build through GitHub Actions after the initial usable implementation was bootstrapped.

The CI gate installs dependencies, copies the bundled Basic Pitch model and TensorFlow.js WASM binaries, and runs the production build.
