# Q&A

## How chat works

Follow-up from last night on "how chat works."

I'm pretty sure Palia used [Vivox](https://docs.unity.com/en-us/vivox-unreal) as the chat provider in 2024, and they probably still do. For context on the codebase: Palia was written in Unreal Engine for both the game client and the game servers, with backend web services in Rust. I'm hazy on some of these details, but this should clear a few things up.

### Setup

Some setup has to happen before chat works:

1. The player's client logs in, which triggers the backend to log in to Vivox. Vivox sends a token back to the client.
2. The client signs in to the Vivox servers to receive messages — a long-lived connection under the hood. On sign-in, it registers for the channels it wants messages on (e.g. `Server15_General`, `Server15_RegionalChat`, `Guild_1234`).

### Sending a message

1. The player hits enter, which makes an API call to the Palia chat backend.
2. The backend censors profanity, then sends the censored message to the Vivox servers on the player's behalf.

### Receiving a message

1. The Vivox API in the receiver's Palia client triggers a callback.
2. (Nintendo Switch only) An additional Nintendo-specific profanity filter runs on-device, required by Nintendo.
3. The Palia Unreal code adds the message to the UI.

### Things I'm not 100% clear on

- **Vivox's own profanity filtering.** Vivox has some, but I don't remember much discussion of it, so I'm not sure why we didn't use it — except that we'd already built our own filter before switching our chat service to Vivox.
- **Token security.** I'm not entirely sure if the game client gets a read-only token from Vivox or a read-write token, though I'm pretty sure the client itself is only reading.