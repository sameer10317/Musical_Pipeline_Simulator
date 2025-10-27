import asyncio
import json
import websockets
from pipeline_simulator import PipelineSimulator, PipelineType, SimulationEvent, EventType
from typing import Set

clients: Set[websockets.WebSocketServerProtocol] = set()
simulator = None
simulation_task = None

async def broadcast_event(event: SimulationEvent):
    if clients:
        message = json.dumps(event.to_dict())
        # Use asyncio.gather safely with currently active clients only
        await asyncio.gather(
            *(client.send(message) for client in list(clients) if not client.closed),
            return_exceptions=True
        )

def event_handler(event: SimulationEvent):
    # Schedule broadcast in event loop thread-safe
    asyncio.get_event_loop().call_soon_threadsafe(asyncio.create_task, broadcast_event(event))

async def run_simulation():
    global simulator
    if simulator and simulator.running:
        await simulator.run_async()

async def handle_client(websocket):  # No 'path' arg for modern websockets, add if needed
    global simulator, simulation_task
    clients.add(websocket)
    print(f"Client connected. Total clients: {len(clients)}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')
                params = data.get('params', {})
                print(f"Received command: {command} with params: {params}")

                if command == 'start':
                    pipeline_type_str = params.get('pipeline_type', 'classic')
                    pipeline_type = PipelineType.SUPERSCALAR if pipeline_type_str == 'superscalar' else PipelineType.CLASSIC
                    simulator = PipelineSimulator(
                        pipeline_type=pipeline_type,
                        clock_speed=params.get('speed', 0.3),
                        event_callback=event_handler
                    )
                    melody = params.get('melody', 'happy_birthday')
                    simulator.start(melody)
                    if simulation_task:
                        simulation_task.cancel()
                    simulation_task = asyncio.create_task(run_simulation())

                elif command == 'pause':
                    if simulator:
                        simulator.pause()
                        if simulation_task:
                            simulation_task.cancel()

                elif command == 'reset':
                    if simulator:
                        simulator.reset()
                        if simulation_task:
                            simulation_task.cancel()
                        # Send reset state immediately
                        await broadcast_event(SimulationEvent(
                            event_type=EventType.CYCLE_TICK,
                            cycle=0,
                            data=simulator.get_state()
                        ))

                elif command == 'stall':
                    if simulator:
                        simulator.inject_stall()

                elif command == 'flush':
                    if simulator:
                        simulator.inject_flush()

                else:
                    print(f"Unknown command: {command}")

            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
            except Exception as e:
                print(f"Error handling message: {e}")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        clients.discard(websocket)
        print(f"Client removed. Total clients: {len(clients)}")

async def main():
    print("=" * 60)
    print("Interactive Pipeline Orchestra - WebSocket Server")
    print("=" * 60)
    print("Server starting on ws://localhost:8765")
    print("Waiting for connections...\n")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
