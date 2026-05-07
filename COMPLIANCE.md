# Compliance Guide

This project is licensed under `GNU AGPL-3.0-or-later`.

This file is a practical summary for:

- users who redistribute builds
- people packaging mirrors or Docker images
- hosted training or inference services
- modified forks

It is not a substitute for the license text in [`LICENSE`](/H:/lora-rescripts/LICENSE).

## What You May Do

- Use the project privately.
- Modify the project.
- Redistribute original or modified copies.
- Charge money for packaging, hosting, support, or related services.

## What You Must Do

- Keep the original license and copyright notices.
- Clearly disclose when your build is modified or unofficial.
- If you distribute the program, provide the corresponding source code under AGPL-compatible terms.
- If you let remote users interact with a modified version over a network, provide those users an opportunity to get the corresponding source code for that version.
- Do not remove source, license, or attribution notices in a way that misleads users about origin.

## Hosted / Cloud Services

If you deploy a modified version as a hosted trainer, web service, notebook image, or managed container for other people to use, you should assume the AGPL network-source obligation applies.

At minimum, provide:

- the source repository URL for the exact deployed version
- the AGPL license text
- a clear notice that the deployment is modified or unofficial if applicable
- corresponding source for your deployed changes

## Metadata And Attribution

This project may embed source and compliance metadata into exported artifacts.

If you publish modified builds:

- do not falsely present them as official
- do not remove attribution in a misleading way
- do mark the build as modified

## No Official Endorsement

Use of this code does not imply endorsement, partnership, or official approval by the original authors.

If you ship a fork, image, mirror, or hosted service, label it clearly as:

- unofficial
- modified
- community-maintained

when that is the case.
