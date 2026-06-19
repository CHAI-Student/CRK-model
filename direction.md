# Directions

This file is the entry point for external repositories and protocol notes used by the model service.
Detailed contracts are maintained in the linked protocol documents below.

## Service Index

| Service | Repository | Protocol Document | Model Directness |
| --- | --- | --- | --- |
| Camera | [CRK-CAMERA](../../CRK-CAMERA) | [camera.md](docs/protocols/camera.md) | Direct |
| IO Board | [CRK-IO-BOARD](../../CRK-IO-BOARD) | [io-board.md](docs/protocols/io-board.md) | Indirect |
| Payment | [CRK-PAYMENT](../../CRK-PAYMENT) | [payment.md](docs/protocols/payment.md) | Indirect |
| Node | [Edge_Environment](../../Edge_Environment) | [node.md](docs/protocols/node.md) | Direct |

## Reading Order

1. [Node protocol](docs/protocols/node.md)
2. [Camera protocol](docs/protocols/camera.md)
3. [IO Board protocol](docs/protocols/io-board.md)
4. [Payment protocol](docs/protocols/payment.md)

## Notes

- The model service directly exchanges API calls with Node and Camera.
- IO Board and Payment are indirect from the model point of view.
- Treat the linked source files in each protocol document as the source of truth.
