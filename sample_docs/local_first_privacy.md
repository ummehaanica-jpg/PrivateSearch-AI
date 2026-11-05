# Local-First Software and Privacy

Local-first software keeps the user's data on the user's own device. The application works without an internet connection, and the user stays in control of who can read their files.

## Why it matters

Sending personal documents to a cloud service creates risks. The provider may store the data for a long time, employees may be able to read it, and a breach can expose everything at once. Keeping documents on a laptop removes these risks, since there is no remote copy to leak.

## On-device machine learning

Models can now run on ordinary laptops. Small language models served by tools such as Ollama answer questions without any network access, and compact embedding models turn text into vectors on the CPU. This makes private search and private question answering practical.

## Data minimisation

Collect only what is needed, keep it only as long as needed, and let the user delete it. A good local application stores its index in a single folder, offers a clear delete button, and never sends telemetry.

## Threats that remain

Local storage is not automatically safe. A stolen laptop exposes its files unless the disk is encrypted, and malware on the device can read local data. Full disk encryption and regular system updates are the simplest protections.
