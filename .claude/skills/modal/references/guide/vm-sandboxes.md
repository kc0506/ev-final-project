# VM Sandboxes

<Callout variant="alpha">

Access to this feature is restricted to allowlisted workspaces only. Compatibility with [Sandbox Snapshots](/docs/guide/sandbox-snapshots) is partial: [Filesystem Snapshots](/docs/guide/sandbox-snapshots#filesystem-snapshots) are supported, but [Memory Snapshots](/docs/guide/sandbox-snapshots#memory-snapshots) are not yet supported.

</Callout>

Sandboxes can be run on top of a full virtual machine rather than on top of gVisor. This gives
each Sandbox a real Linux kernel, which makes certain workloads (e.g. Docker systems) behave
the way they would on a normal Linux host.

You can use the VM runtime for your Sandbox by passing `experimental_options={"vm_runtime": True}`
to `Sandbox.create()`.

```python fixture:sb_app
with modal.enable_output():
    sb = modal.Sandbox.create(
        app=sb_app,
        cpu=2,  # physical cores
        memory=4096,  # MiB
        experimental_options={"vm_runtime": True},
    )

# add a script that uses VM Sandbox features
sb.filesystem.write_text(
    """
  # Format an ext4 filesystem onto a regular file.
  truncate -s 100M /tmp/disk.img
  mkfs.ext4 -F /tmp/disk.img

  # Mount it. This works in a VM, but isn't supported in gVisor.
  mkdir -p /mnt/loop
  mount -o loop /tmp/disk.img /mnt/loop
""",
    "/tmp/mount_loopback_filesystem.sh",
)

p = sb.exec("bash", "/tmp/mount_loopback_filesystem.sh")
p.wait()

print(p.stdout.read())

print(p.stderr.read())

assert p.returncode == 0  # error if the program in the Sandbox fails

sb.terminate()
```

Additionally, quickly provision a VM runtime sandbox with a PTY shell via the CLI using:

```
modal shell --experimental-option vm_runtime=1
```

## Improvements over gVisor sandboxes

Docker workloads behave more like they do in a non-container environment. In particular:

* Docker state (e.g. `/var/lib/docker`) is included in [Filesystem Snapshots](/docs/guide/sandbox-snapshots#filesystem-snapshots)
* Docker features that previously needed special treatment on gVisor (e.g. inter-container networking) will also work normally

Features that only make sense in a bona fide Linux environment are now available:

* Custom [init systems](https://arxiv.org/pdf/0706.2748) (such as [`systemd`](https://man7.org/linux/man-pages/man1/systemd.1.html)) are supported
* [eBPF](https://ebpf.io/) is supported
* Resource isolation within the Sandbox via [cgroups](https://man7.org/linux/man-pages/man7/cgroups.7.html) is supported

Finally, for most workloads, the root filesystem will perform better on a VM Sandbox than in a gVisor Sandbox. The benefit is less pronounced for metadata-heavy workloads.

## Resource model

Unlike [resource provisioning](/docs/guide/resources) in other runtimes,
memory provisioning is **static** for VM Sandboxes: you get exactly as much
RAM as you request via `memory` argument to `Sandbox.create`. By default, VM
sandboxes get 1GiB of RAM.

However, CPU provisioning is elastic. You can burst above your requested amount.

Costs for both resources are calculated based on the requested amount, used amount,
the duration of Sandbox execution, and [our rates for `cpu` and `memory`](/pricing).

## Limitations

The following limitations are known and we're tracking them:

* **GPUs are not supported.** VM Sandboxes currently only support CPU workloads.
* **`setuid` bits are not faithfully preserved in Modal [Images](/docs/guide/images).**
  * Executables that rely on the `setuid` bit (e.g. `sudo`, `ping`, `mount`) may not work properly.
  * This can be worked around if it matters for your workload: re-apply the `setuid` bit at runtime as the `root` user (e.g. by running `chmod u+s /path/to/binary`).
* **The [Sandbox filesystem API](/docs/guide/sandbox-files#filesystem-api-beta) is only available in new SDK versions**. For the Python SDK, it requires version ≥ 1.4.0 and for the JS/TS/Go SDKs, it requires versions ≥ 0.7.6.
* **[`Sandbox.reload_volumes()`](/docs/reference/modal.Sandbox#reload_volumes) is not supported.** VM Sandboxes do not currently support reloading volumes at runtime.
* **[Memory Snapshots](/docs/guide/sandbox-snapshots#memory-snapshots) are not yet supported.** Only
  [Filesystem Snapshots](/docs/guide/sandbox-snapshots#filesystem-snapshots) work on VM Sandboxes today.
* **Root images ≥ 512 GiB are not supported.** The VM root filesystem is currently limited to 512 GiB. Sandboxes created from container images exceeding this size will fail to start.

If you hit a rough edge that isn't listed here, please reach out via [Slack](/slack) or email us at <support@modal.com>.
