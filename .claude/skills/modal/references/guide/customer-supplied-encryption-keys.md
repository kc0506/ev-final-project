# Customer Supplied Encryption Keys

<Callout variant="alpha" />

Customer Supplied Encryption Keys (CSEK) let you provide your own key material
when creating supported Modal resources. Modal uses the key as part of the
resource encryption flow, but does not persist the key.

Use CSEK when you need customer-held key material for data that Modal stores.
You are responsible for generating, storing, backing up, and providing the key
again when the protected resource is used.

<Callout variant="warning">

If you lose the CSEK for a protected resource, Modal cannot recover it for you.
Store the key in a durable key management system outside Modal.

</Callout>

## How CSEK works

For each supported resource, the same basic flow applies:

1. Generate key material with a cryptographically secure random source.
2. Pass the key when creating the resource.
3. Store the resource ID and key in your own system.
4. Pass the same key again when reading, mounting, or restoring the resource.

The key must be passed as `bytes`, must not be empty, and must be between 16 and
512 bytes long.

```python notest
import secrets

encryption_key = secrets.token_bytes(32)
```

Do not commit CSEK material to source control, bake it into Images, print it in
logs, or store it next to the data it protects. Prefer a dedicated key
management system or secrets manager with access controls and backup policies
that match your security requirements.

## Supported resources

This page documents CSEK for the currently supported resources. As CSEK support
becomes available for more Modal resources, additional sections will be added.

| Resource                                    | SDK support |
| ------------------------------------------- | ----------- |
| [Directory Snapshots](#directory-snapshots) | Python SDK  |

## Directory Snapshots

[Directory Snapshots](/docs/guide/sandbox-snapshots#directory-snapshots) let
you capture a directory from a running Sandbox as an
[Image](/docs/reference/modal.Image). With CSEK, you pass key material when
creating the snapshot and pass the same key again when mounting it.

### Create a CSEK-protected directory snapshot

```python notest
import secrets

import modal

app = modal.App.lookup("csek-directory-snapshots", create_if_missing=True)
encryption_key = secrets.token_bytes(32)

sb = modal.Sandbox.create(app=app)
sb.exec(
    "bash",
    "-c",
    "mkdir -p /project && echo 'private data' > /project/state.txt",
).wait()

snapshot = sb.snapshot_directory(
    "/project",
    _experimental_encryption_key=encryption_key,
)
sb.terminate()

# Store both values in your own durable systems.
snapshot_id = snapshot.object_id
```

The `_experimental_encryption_key` parameter is currently exposed as an
experimental Python SDK API. See [Feature maturity](/docs/guide/feature-maturity#experimental-sdk)
for how Modal treats experimental SDK surfaces.

### Mount a CSEK-protected directory snapshot

To use the snapshot later, rehydrate the Image by ID and pass the same key to
`Sandbox.mount_image()`.

```python notest
import modal

app = modal.App.lookup("csek-directory-snapshots")
snapshot = modal.Image.from_id(snapshot_id)

sb = modal.Sandbox.create(app=app)
sb.mount_image(
    "/project",
    snapshot,
    _experimental_encryption_key=encryption_key,
)

contents = sb.exec("cat", "/project/state.txt").stdout.read().strip()
assert contents == "private data"
sb.terminate()
```

If the key is missing or incorrect, Modal cannot mount the encrypted snapshot.

### Re-snapshotting encrypted directories

After mounting a CSEK-protected directory snapshot, you can create another
directory snapshot from that mounted path:

* Pass `_experimental_encryption_key` to protect the new snapshot with CSEK.
* Omit `_experimental_encryption_key` to create the new snapshot with
  Modal-managed encryption.

Each CSEK-protected snapshot is tied to the key used when that snapshot was
created. If you create a new CSEK-protected snapshot with a different key, use
the new key when mounting the new snapshot.

### Retention

CSEK does not change Directory Snapshot retention. Directory Snapshots are
retained for 30 days after creation. See [Snapshot Retention](/docs/guide/sandbox-snapshots#snapshot-retention)
for details.
