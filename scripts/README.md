# Scripts

This folder contains several convenience scripts and executables to streamline hermes development and distribution.

## Updating GitLab test pipeline Docker images

Changes in requirements need to be reflected in updated Docker test pipeline images.
Run `./docker/ci-build-images.sh` to build the latest images locally.

Before pushing to the remote registry, the newly built images can be tested by spinning them up,
building HermesPy within them and running all unit tests via `./docker/ci-test-images.sh`.

Afterwards, push the images to the GitLab registry via `./docker/ci-push-images.sh`.
Note that the login requires an access token to be generated via the GitLab interface.