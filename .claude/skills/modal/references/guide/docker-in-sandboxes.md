# Docker in Sandboxes

<Callout variant="alpha"></Callout>

Modal has Alpha support for running Docker containers inside `modal.Sandbox`.
This is intended to support coding agents who want to interact with development environments that are managed using Docker. We recommend that you use [VM sandboxes](/docs/guide/vm-sandboxes) for such workloads.

## Demo

Copy the following program to e.g. `docker_in_modal_demo.py`, and run it with `python docker_in_modal_demo.py`.
Note that we use a VM sandbox in this example.

```python
import modal

# Create an image for the parent Modal Sandbox, with Docker installed.
def create_modal_sandbox_image():
    image = (
        modal.Image.from_registry("ubuntu:24.04")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install(["docker.io", "docker-buildx"])
        .run_commands("mkdir /build")
    )
    return image


def main():
    print("Looking up modal.Sandbox app")
    app = modal.App.lookup("docker-test", create_if_missing=True)
    print("Creating sandbox")

    with modal.enable_output():
        sb = modal.Sandbox.create(
            "/usr/bin/dockerd",
            "-D",
            timeout=60 * 60,
            app=app,
            image=create_modal_sandbox_image(),
            experimental_options={"vm_runtime": True},
        )

    print(f"sandbox_id: {sb.object_id}")
    task_id = sb._get_task_id()
    print(f"task_id: {task_id}")
    print(f"To shell into the task, run: modal shell {task_id}")
    # dockerd is the sandbox entrypoint and takes a moment to bind
    # /var/run/docker.sock after the sandbox is created. Poll until the
    # daemon answers so the first `docker build` doesn't run before dockerd is ready.
    print("Waiting for dockerd to be ready")
    wait_p = sb.exec(
        "sh",
        "-c",
        "for i in $(seq 1 120); do "
        "if [ -S /var/run/docker.sock ] && docker info >/dev/null 2>&1; then "
        "echo ready; exit 0; fi; sleep 1; done; "
        "echo 'dockerd not ready after 120s' >&2; exit 1",
    )
    wait_p.wait()
    if wait_p.returncode != 0:
        raise Exception(f"dockerd never became ready: {wait_p.stderr.read()}")

    # A simple Dockerfile that we'll build and run within Modal.
    dockerfile = """
    FROM ubuntu
    RUN apt-get update
    RUN apt-get install -y cowsay curl
    RUN mkdir -p /usr/share/cowsay/cows/
    RUN curl -o /usr/share/cowsay/cows/docker.cow https://raw.githubusercontent.com/docker/whalesay/master/docker.cow
    ENTRYPOINT ["/usr/games/cowsay", "-f", "docker.cow"]
    """
    sb.filesystem.write_text(dockerfile, "/build/Dockerfile")

    print("Building docker image")
    p = sb.exec("docker", "build", "--network=host", "-t", "whalesay", "/build")
    for l in p.stdout:
        print(l, end="")
    p.wait()
    print("--------------------------------")
    if p.returncode != 0:
        print(p.stderr.read())
        raise Exception("Docker build failed")

    # The Sandbox will run a container from the built image and print this:
    #
    #  ________
    # < Hello! >
    #  --------
    #     \
    #      \
    #       \
    #                     ##         .
    #               ## ## ##        ==
    #            ## ## ## ## ##    ===
    #        /"""""""""""""""""\___/ ===
    #       {                       /  ===-
    #        \______ O           __/
    #          \    \         __/
    #           \____\_______/

    print("Running Docker image")
    # Note we can't use -it here because we're not in a TTY.
    p = sb.exec("docker", "run", "--rm", "whalesay", "Hello!")
    print(p.stdout.read())
    p.wait()
    if p.returncode != 0:
        raise Exception(f"Docker run failed: {p.stderr.read()}")
    sb.terminate()

if __name__ == "__main__":
    main()
```
