# The MIT License (MIT)
# Copyright © 2025 Quant by OpenGradient

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import os
import time
import typing

import requests
import bittensor as bt

# Bittensor Miner Quant:
import quant

# import base miner class which takes care of most of the boilerplate
from quant.base.miner import BaseMinerNeuron

# Import the shared Quant agent server module
from neurons import quant_agent_server

_AGENT_QUERY_URL: typing.Optional[str] = None


def _get_agent_query_url() -> str:
    """Construct the Quant agent query URL."""

    global _AGENT_QUERY_URL

    if _AGENT_QUERY_URL is None:
        base_url = os.getenv("QUANT_AGENT_SERVER_URL")
        if base_url:
            base = base_url.rstrip("/")
        else:
            base = f"http://127.0.0.1:{quant_agent_server.QUANT_AGENT_SERVER_PORT}"

        route = os.getenv("QUANT_AGENT_QUERY_ROUTE", "/query")
        if not route.startswith("/"):
            route = f"/{route}"

        _AGENT_QUERY_URL = f"{base}{route}"

    return _AGENT_QUERY_URL


def _get_agent_timeout() -> float:
    """Fetch the request timeout for the Quant agent server."""

    timeout_value = os.getenv("QUANT_AGENT_TIMEOUT", "60")
    try:
        return float(timeout_value)
    except (TypeError, ValueError):
        bt.logging.warning(
            "Invalid QUANT_AGENT_TIMEOUT value '%s'. Falling back to 60 seconds.",
            timeout_value,
        )
        return 60.0


def _coerce_bytes(value: typing.Any) -> bytes:
    """Ensure a field from the agent payload is converted to bytes."""

    if isinstance(value, bytes):
        return value
    if value is None:
        return b""
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()

    try:
        return bytes(value)
    except (TypeError, ValueError):
        return str(value).encode("utf-8")


def _normalize_proofs(proofs: typing.Any) -> list[bytes]:
    """Convert the proofs collection into a list of bytes."""

    if not isinstance(proofs, (list, tuple)):
        return []

    normalized: list[bytes] = []
    for proof in proofs:
        normalized.append(_coerce_bytes(proof))
    return normalized


def _extract_quant_response_payload(
    payload: typing.Any,
) -> dict[str, typing.Any]:
    """Extract the QuantResponse payload from the agent JSON response."""

    if isinstance(payload, dict):
        for key in ("data", "result", "payload"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        return payload

    return {"response": payload}


def _build_quant_response(
    payload: dict[str, typing.Any],
    query: quant.protocol.QuantQuery,
) -> quant.protocol.QuantResponse:
    """Construct a QuantResponse object from the agent payload."""

    response_text = payload.get("response")
    if not isinstance(response_text, str):
        response_text = "" if response_text is None else str(response_text)

    signature = _coerce_bytes(payload.get("signature"))
    proofs = _normalize_proofs(payload.get("proofs"))

    metadata_field = payload.get("metadata")
    if isinstance(metadata_field, dict):
        metadata = dict(metadata_field)
    else:
        metadata = {}
        if metadata_field is not None:
            metadata["raw_metadata"] = metadata_field

    metadata.setdefault("source", "quant_agent_server")
    if query.metadata:
        metadata.setdefault("query_metadata", query.metadata)

    return quant.protocol.QuantResponse(
        response=response_text,
        signature=signature,
        proofs=proofs,
        metadata=metadata,
    )


def _send_quant_agent_request(
    query: quant.protocol.QuantQuery,
) -> typing.Optional[quant.protocol.QuantResponse]:
    """Perform the HTTP request to the Quant agent server."""

    if query is None:
        bt.logging.error("Quant agent request received an empty query object.")
        return None

    payload = {
        "query": query.query,
        "userID": query.userID,
        "metadata": query.metadata or {},
    }

    response_obj: typing.Optional[requests.Response] = None
    try:
        response_obj = requests.post(
            _get_agent_query_url(),
            json=payload,
            timeout=_get_agent_timeout(),
        )
        response_obj.raise_for_status()
        raw_payload = response_obj.json()
    except requests.RequestException as exc:
        bt.logging.error(f"Quant agent request failed: {exc}")
        return None
    except ValueError as exc:
        bt.logging.error(f"Quant agent returned invalid JSON: {exc}")
        return None
    finally:
        if response_obj is not None:
            response_obj.close()

    try:
        quant_payload = _extract_quant_response_payload(raw_payload)
        return _build_quant_response(quant_payload, query)
    except Exception as exc:  # pylint: disable=broad-except
        bt.logging.error(f"Unable to build QuantResponse from agent payload: {exc}")
        return None


async def query_quant_agent(
    query: quant.protocol.QuantQuery,
) -> typing.Optional[quant.protocol.QuantResponse]:
    """Fetch a QuantResponse from the locally running Quant agent server."""

    try:
        return await asyncio.to_thread(_send_quant_agent_request, query)
    except RuntimeError:
        # Fallback in case we're running outside of an event loop.
        return _send_quant_agent_request(query)

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # TODO(developer): Anything specific to your use case you can do here

    async def forward(
        self, synapse: quant.protocol.QuantSynapse
    ) -> quant.protocol.QuantSynapse:
        """
        Processes the incoming 'QuantSynapse' request by forwarding it to the locally managed
        BitQuant agent server via the `query_quant_agent` helper.

        Args:
            synapse (quant.protocol.QuantSynapse): The synapse object containing the query.

        Returns:
            quant.protocol.QuantSynapse: The synapse object with the response set.
        """
        try:
            # Check if query is properly set
            if not hasattr(synapse, 'query') or synapse.query is None:
                bt.logging.warning("Received synapse with no query")
                response = quant.protocol.QuantResponse(
                    response="Error: No query provided",
                    signature=b"error",
                    proofs=[b"error"],
                    metadata={}
                )
            else:
                bt.logging.info(f"Processing query: {synapse.query.query}")
                bt.logging.info(f"From userID: {synapse.query.userID}")
                bt.logging.info(f"Metadata: {synapse.query.metadata}")
                
                # TODO(developer): Developers deploying miner nodes can add their own custom mining logic here.
                response = await query_quant_agent(synapse.query)
                
                if response is None:
                    bt.logging.warning("Quant agent returned no response; providing fallback error message.")
                    response = quant.protocol.QuantResponse(
                        response="Error: Failed to communicate with quant agent",
                        signature=b"error",
                        proofs=[b"error"],
                        metadata=synapse.query.metadata or {}
                    )
                else:
                    bt.logging.info(f"Received response from quant agent: {response.response[:100]}...")

                response_metadata = dict(response.metadata or {})
                response_metadata["miner_id"] = self.wallet.hotkey.ss58_address
                response.metadata = response_metadata
                synapse.response = {
                    "response": response.response,
                    "signature": response.signature,
                    "proofs": response.proofs,
                    "metadata": response_metadata,
                }
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            # Provide a fallback response in case of error
            error_response = {
                "response": f"Error processing request: {str(e)}",
                "signature": b"error",
                "proofs": [b"error"],
                "metadata": {}
            }
            # Set the response as a dictionary
            synapse.response = error_response
            return synapse

    async def blacklist(
        self, synapse: quant.protocol.QuantSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(
        self, synapse: quant.protocol.QuantSynapse
    ) -> float:
        """
        The priority function determines the order in which requests are handled.
        """
        # Get the dendrite.hotkey from axon_info if it exists
        hotkey = None
        if hasattr(synapse, 'dendrite') and synapse.dendrite is not None:
            hotkey = synapse.dendrite.hotkey
        elif hasattr(synapse, 'axon_info') and synapse.axon_info is not None:
            hotkey = synapse.axon_info.hotkey
            
        if hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        try:
            caller_uid = self.metagraph.hotkeys.index(hotkey)
            priority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        except (ValueError, KeyError, IndexError) as e:
            # If the hotkey is not in the metagraph or any other indexing error, return 0 priority
            bt.logging.trace(f"Error getting priority for {hotkey}: {e}")
            priority = 0.0
            
        bt.logging.trace(
            f"Prioritizing {hotkey} with value: {priority}"
        )
        return priority

# This is the main function, which runs the miner.
if __name__ == "__main__":
    try:
        # Set up and start the Quant agent server using the shared module
        quant_agent_server_process = quant_agent_server.setup_quant_agent_server()
        
        # Log whether we're using an existing server or started a new one
        if quant_agent_server_process is None:
            bt.logging.info("Using an existing Quant agent server or continuing without one.")
        else:
            bt.logging.info(f"Started a new Quant agent server with PID: {quant_agent_server_process.pid}")
            
        with Miner() as miner:
            bt.logging.info("Starting miner...")
            while True:
                bt.logging.info(f"Miner running... {time.time()}")
                
                # Check if Quant agent server is still running (only if we're managing it)
                if quant_agent_server_process is not None and not quant_agent_server.check_quant_agent_server(quant_agent_server_process):
                    bt.logging.error("Quant agent server is no longer running.")
                    bt.logging.warning("Continuing miner operation, but queries may fail.")
                    # Set to None so we don't keep checking
                    quant_agent_server_process = None
                
                time.sleep(5)
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt received. Exiting miner.")
    except Exception as e:
        bt.logging.error(f"Error running miner: {e}")
        import traceback
        bt.logging.debug(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        # The cleanup will be handled by the atexit handler in the shared module.
        # Since we registered the atexit handler in quant_agent_server, it will only clean up
        # the server if it was started by this process.
        pass
