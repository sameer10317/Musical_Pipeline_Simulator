"""
Interactive Pipeline Orchestra - Backend Simulation Module

This module simulates CPU pipeline architectures with musical note events.
It models instruction flow through different pipeline types, handles hazards,
and provides an API interface for front-end integration.

Author: Pipeline Orchestra Project
Date: 2025
"""

import asyncio
import json
import time
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable, Any
from collections import deque
import threading


# ========== Enumerations ==========

class PipelineType(Enum):
    """Supported pipeline architectures"""
    CLASSIC = "classic"  # Classic 5-stage pipeline
    SUPERSCALAR = "superscalar"  # 2-way superscalar pipeline


class InstructionState(Enum):
    """Possible states for an instruction"""
    FETCHING = "IF"
    DECODING = "ID"
    EXECUTING = "EX"
    MEMORY = "MEM"
    WRITEBACK = "WB"
    COMPLETED = "COMPLETED"
    FLUSHED = "FLUSHED"


class EventType(Enum):
    """Types of events emitted by the simulator"""
    INSTRUCTION_MOVED = "instruction_moved"
    NOTE_PLAY = "note_play"
    STALL_INJECTED = "stall_injected"
    FLUSH_INJECTED = "flush_injected"
    CYCLE_TICK = "cycle_tick"
    SIMULATION_COMPLETE = "simulation_complete"
    INSTRUCTION_COMPLETED = "instruction_completed"


# ========== Data Classes ==========

@dataclass
class Instruction:
    """Represents a single instruction (musical note) in the pipeline"""
    id: int
    note: str  # Musical note (e.g., 'C4', 'D4')
    name: str  # Display name (e.g., 'Happy', 'Birth-day')
    state: InstructionState = InstructionState.FETCHING
    stalled: bool = False
    pipeline_id: int = 0  # Which pipeline this instruction is in (for superscalar)
    stage_index: int = 0  # Current stage index (0-4)
    
    def to_dict(self) -> Dict:
        """Convert instruction to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'note': self.note,
            'name': self.name,
            'state': self.state.value,
            'stalled': self.stalled,
            'pipeline_id': self.pipeline_id,
            'stage_index': self.stage_index
        }


@dataclass
class SimulationEvent:
    """Event emitted during simulation"""
    event_type: EventType
    cycle: int
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary for JSON serialization"""
        return {
            'event_type': self.event_type.value,
            'cycle': self.cycle,
            'data': self.data,
            'timestamp': self.timestamp
        }


@dataclass
class PipelineStatistics:
    """Statistics about pipeline execution"""
    total_cycles: int = 0
    instructions_completed: int = 0
    total_stalls: int = 0
    total_flushes: int = 0
    
    def efficiency(self) -> float:
        """Calculate pipeline efficiency (ideal cycles / actual cycles)"""
        if self.total_cycles == 0:
            return 100.0
        return (self.instructions_completed / self.total_cycles) * 100
    
    def to_dict(self) -> Dict:
        """Convert statistics to dictionary"""
        return {
            'total_cycles': self.total_cycles,
            'instructions_completed': self.instructions_completed,
            'total_stalls': self.total_stalls,
            'total_flushes': self.total_flushes,
            'efficiency': round(self.efficiency(), 2)
        }


# ========== Pipeline Stage Definitions ==========

class PipelineStage:
    """Represents a single stage in the pipeline"""
    
    STAGES = [
        InstructionState.FETCHING,
        InstructionState.DECODING,
        InstructionState.EXECUTING,
        InstructionState.MEMORY,
        InstructionState.WRITEBACK
    ]
    
    STAGE_NAMES = {
        InstructionState.FETCHING: "Instruction Fetch",
        InstructionState.DECODING: "Instruction Decode",
        InstructionState.EXECUTING: "Execute",
        InstructionState.MEMORY: "Memory Access",
        InstructionState.WRITEBACK: "Write Back"
    }


# ========== Melody Definitions ==========

class MelodyLibrary:
    """Library of predefined melodies"""
    
    MELODIES = {
        'happy_birthday': [
            # Happy birthday to you
            {'note': 'G4', 'name': 'Hap-', 'duration': 0.3},
            {'note': 'G4', 'name': '-py', 'duration': 0.3},
            {'note': 'A4', 'name': 'birth-', 'duration': 0.6},
            {'note': 'G4', 'name': '-day', 'duration': 0.6},
            {'note': 'C5', 'name': 'to', 'duration': 0.6},
            {'note': 'B4', 'name': 'you', 'duration': 1.2},
            
            # Happy birthday to you
            {'note': 'G4', 'name': 'Hap-', 'duration': 0.3},
            {'note': 'G4', 'name': '-py', 'duration': 0.3},
            {'note': 'A4', 'name': 'birth-', 'duration': 0.6},
            {'note': 'G4', 'name': '-day', 'duration': 0.6},
            {'note': 'D5', 'name': 'to', 'duration': 0.6},
            {'note': 'C5', 'name': 'you', 'duration': 1.2},
            
            # Happy birthday to my dear love
            {'note': 'G4', 'name': 'Hap-', 'duration': 0.3},
            {'note': 'G4', 'name': '-py', 'duration': 0.3},
            {'note': 'G5', 'name': 'birth-', 'duration': 0.6},
            {'note': 'E5', 'name': '-day', 'duration': 0.6},
            {'note': 'C5', 'name': 'to', 'duration': 0.6},
            {'note': 'B4', 'name': 'my', 'duration': 0.6},
            {'note': 'A4', 'name': 'dear', 'duration': 0.6},
            {'note': 'A4', 'name': 'love', 'duration': 1.2},
            
            # Happy birthday to you
            {'note': 'F5', 'name': 'Hap-', 'duration': 0.3},
            {'note': 'F5', 'name': '-py', 'duration': 0.3},
            {'note': 'E5', 'name': 'birth-', 'duration': 0.6},
            {'note': 'C5', 'name': '-day', 'duration': 0.6},
            {'note': 'D5', 'name': 'to', 'duration': 0.6},
            {'note': 'C5', 'name': 'you', 'duration': 1.2},
            
            # May God bless you
            {'note': 'G4', 'name': 'May', 'duration': 0.4},
            {'note': 'C5', 'name': 'God', 'duration': 0.6},
            {'note': 'B4', 'name': 'bless', 'duration': 0.8},
            {'note': 'C5', 'name': 'you', 'duration': 1.6},
        ],
        'twinkle': [
            {'note': 'C4', 'name': 'Note 1', 'duration': 0.5},
            {'note': 'C4', 'name': 'Note 2', 'duration': 0.5},
            {'note': 'G4', 'name': 'Note 3', 'duration': 0.5},
            {'note': 'G4', 'name': 'Note 4', 'duration': 0.5},
            {'note': 'A4', 'name': 'Note 5', 'duration': 0.5},
            {'note': 'A4', 'name': 'Note 6', 'duration': 0.5},
            {'note': 'G4', 'name': 'Note 7', 'duration': 1.0},
            {'note': 'F4', 'name': 'Note 8', 'duration': 0.5},
            {'note': 'F4', 'name': 'Note 9', 'duration': 0.5},
            {'note': 'E4', 'name': 'Note 10', 'duration': 0.5},
            {'note': 'E4', 'name': 'Note 11', 'duration': 0.5},
            {'note': 'D4', 'name': 'Note 12', 'duration': 0.5},
        ],
        'scale': [
            {'note': 'C4', 'name': 'Do', 'duration': 0.5},
            {'note': 'D4', 'name': 'Re', 'duration': 0.5},
            {'note': 'E4', 'name': 'Mi', 'duration': 0.5},
            {'note': 'F4', 'name': 'Fa', 'duration': 0.5},
            {'note': 'G4', 'name': 'Sol', 'duration': 0.5},
            {'note': 'A4', 'name': 'La', 'duration': 0.5},
            {'note': 'B4', 'name': 'Ti', 'duration': 0.5},
            {'note': 'C5', 'name': 'Do', 'duration': 1.0},
        ]
    }
    
    @classmethod
    def get_melody(cls, melody_name: str) -> List[Dict]:
        """Get a melody by name"""
        return cls.MELODIES.get(melody_name, cls.MELODIES['happy_birthday'])


# ========== Main Pipeline Simulator ==========

class PipelineSimulator:
    """
    Main pipeline simulator class.
    
    This class simulates instruction flow through CPU pipelines,
    handles hazards (stalls and flushes), and emits events for
    front-end visualization and audio playback.
    """
    
    def __init__(
        self,
        pipeline_type: PipelineType = PipelineType.CLASSIC,
        clock_speed: float = 0.3,
        event_callback: Optional[Callable[[SimulationEvent], None]] = None
    ):
        """
        Initialize the pipeline simulator.
        
        Args:
            pipeline_type: Type of pipeline architecture to simulate
            clock_speed: Time between clock cycles in seconds
            event_callback: Callback function to receive simulation events
        """
        self.pipeline_type = pipeline_type
        self.clock_speed = clock_speed
        self.event_callback = event_callback
        
        # Simulation state
        self.running = False
        self.paused = False
        self.cycle_count = 0
        
        # Pipeline configuration
        self.num_pipelines = 2 if pipeline_type == PipelineType.SUPERSCALAR else 1
        self.num_stages = 5  # Always 5 stages for both architectures
        
        # Pipeline state: List of pipelines, each containing stages (list of instructions)
        # pipelines[pipeline_id][stage_index] = Instruction or None
        self.pipelines: List[List[Optional[Instruction]]] = [
            [None] * self.num_stages for _ in range(self.num_pipelines)
        ]
        
        # Instruction queue
        self.instruction_queue: deque = deque()
        self.instruction_counter = 0
        
        # Statistics
        self.stats = PipelineStatistics()
        
        # For async operation
        self._task: Optional[asyncio.Task] = None
        
        # Hazard flags
        self.stall_next_cycle = False
        self.flush_next_cycle = False
    
    def load_melody(self, melody_name: str):
        """
        Load a melody into the instruction queue.
        
        Args:
            melody_name: Name of the melody to load
        """
        melody_data = MelodyLibrary.get_melody(melody_name)
        self.instruction_queue.clear()
        self.instruction_counter = 0
        
        for note_data in melody_data:
            self.instruction_queue.append(note_data)
    
    def set_clock_speed(self, speed: float):
        """Set the clock speed (seconds per cycle)"""
        self.clock_speed = max(0.05, min(2.0, speed))  # Clamp between 0.05 and 2.0
    
    def start(self, melody_name: str = 'happy_birthday'):
        """
        Start the simulation.
        
        Args:
            melody_name: Name of melody to play
        """
        if self.running and not self.paused:
            return
        
        if not self.paused:
            # Fresh start
            self.reset()
            self.load_melody(melody_name)
        
        self.running = True
        self.paused = False
    
    def pause(self):
        """Pause the simulation"""
        if self.running:
            self.paused = True
            self.running = False
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.running = False
        self.paused = False
        self.cycle_count = 0
        self.instruction_counter = 0
        
        # Clear pipelines
        for p in range(self.num_pipelines):
            for s in range(self.num_stages):
                self.pipelines[p][s] = None
        
        # Clear queue
        self.instruction_queue.clear()
        
        # Reset statistics
        self.stats = PipelineStatistics()
        
        # Reset hazard flags
        self.stall_next_cycle = False
        self.flush_next_cycle = False
    
    def inject_stall(self):
        """Inject a pipeline stall for the next cycle"""
        if self.running and not self.paused:
            self.stall_next_cycle = True
            self.stats.total_stalls += 1
    
    def inject_flush(self):
        """Inject a pipeline flush for the next cycle"""
        if self.running and not self.paused:
            self.flush_next_cycle = True
            self.stats.total_flushes += 1
    
    def _emit_event(self, event: SimulationEvent):
        """Emit an event to the callback"""
        if self.event_callback:
            self.event_callback(event)
    
    def _fetch_instruction(self, pipeline_id: int) -> Optional[Instruction]:
        """
        Fetch a new instruction for the given pipeline.
        
        Args:
            pipeline_id: Which pipeline to fetch for
            
        Returns:
            New instruction or None if queue is empty
        """
        if not self.instruction_queue:
            return None
        
        note_data = self.instruction_queue.popleft()
        instruction = Instruction(
            id=self.instruction_counter,
            note=note_data['note'],
            name=note_data['name'],
            state=InstructionState.FETCHING,
            pipeline_id=pipeline_id,
            stage_index=0
        )
        # Store duration in instruction object
        instruction.duration = note_data.get('duration', 0.5)
        self.instruction_counter += 1
        
        return instruction
    
    def _advance_pipeline(self, pipeline_id: int):
        """
        Advance instructions in a single pipeline by one stage.
        
        Args:
            pipeline_id: Index of the pipeline to advance
        """
        pipeline = self.pipelines[pipeline_id]
        
        # Process stages from back to front to avoid overwriting
        for stage_idx in range(self.num_stages - 1, -1, -1):
            instruction = pipeline[stage_idx]
            
            if instruction is None:
                continue
            
            # Skip if instruction is stalled
            if instruction.stalled:
                instruction.stalled = False  # Clear stall for next cycle
                continue
            
            # Last stage: Complete the instruction
            if stage_idx == self.num_stages - 1:
                instruction.state = InstructionState.COMPLETED
                self.stats.instructions_completed += 1
                
                self._emit_event(SimulationEvent(
                    event_type=EventType.INSTRUCTION_COMPLETED,
                    cycle=self.cycle_count,
                    data={'instruction': instruction.to_dict()}
                ))
                
                pipeline[stage_idx] = None
            else:
                # Move to next stage if it's empty
                if pipeline[stage_idx + 1] is None:
                    pipeline[stage_idx + 1] = instruction
                    pipeline[stage_idx] = None
                    
                    instruction.stage_index = stage_idx + 1
                    instruction.state = PipelineStage.STAGES[stage_idx + 1]
                    
                    # Emit movement event
                    self._emit_event(SimulationEvent(
                        event_type=EventType.INSTRUCTION_MOVED,
                        cycle=self.cycle_count,
                        data={
                            'instruction': instruction.to_dict(),
                            'from_stage': stage_idx,
                            'to_stage': stage_idx + 1
                        }
                    ))
                    
                    # If moving to execute stage, emit note play event
                    if stage_idx + 1 == 2:  # Execute stage
                        # Get duration from instruction data (default 0.5)
                        duration = getattr(instruction, 'duration', 0.5)
                        self._emit_event(SimulationEvent(
                            event_type=EventType.NOTE_PLAY,
                            cycle=self.cycle_count,
                            data={
                                'note': instruction.note,
                                'instruction_id': instruction.id,
                                'pipeline_id': pipeline_id,
                                'duration': duration
                            }
                        ))
        
        # Fetch new instruction if first stage is empty
        if pipeline[0] is None:
            new_instruction = self._fetch_instruction(pipeline_id)
            if new_instruction:
                pipeline[0] = new_instruction
                
                self._emit_event(SimulationEvent(
                    event_type=EventType.INSTRUCTION_MOVED,
                    cycle=self.cycle_count,
                    data={
                        'instruction': new_instruction.to_dict(),
                        'from_stage': -1,
                        'to_stage': 0
                    }
                ))
    
    def _handle_stall(self):
        """Apply stall to all instructions in all pipelines"""
        for pipeline in self.pipelines:
            for instruction in pipeline:
                if instruction:
                    instruction.stalled = True
        
        self._emit_event(SimulationEvent(
            event_type=EventType.STALL_INJECTED,
            cycle=self.cycle_count,
            data={'message': 'Pipeline stalled for one cycle'}
        ))
    
    def _handle_flush(self):
        """Flush all instructions from all pipelines"""
        flushed_instructions = []
        
        for pipeline_id, pipeline in enumerate(self.pipelines):
            for stage_idx, instruction in enumerate(pipeline):
                if instruction:
                    instruction.state = InstructionState.FLUSHED
                    flushed_instructions.append(instruction.to_dict())
                    pipeline[stage_idx] = None
        
        self._emit_event(SimulationEvent(
            event_type=EventType.FLUSH_INJECTED,
            cycle=self.cycle_count,
            data={
                'message': 'Pipeline flushed',
                'flushed_instructions': flushed_instructions
            }
        ))
    
    def _is_complete(self) -> bool:
        """Check if simulation is complete"""
        # Check if queue is empty
        if self.instruction_queue:
            return False
        
        # Check if all pipelines are empty
        for pipeline in self.pipelines:
            if any(stage is not None for stage in pipeline):
                return False
        
        return True
    
    def tick(self):
        """
        Execute one clock cycle of the simulation.
        This is the main simulation logic executed each cycle.
        """
        self.cycle_count += 1
        self.stats.total_cycles += 1
        
        # Handle hazards first
        if self.flush_next_cycle:
            self._handle_flush()
            self.flush_next_cycle = False
            # After flush, skip normal pipeline advancement
            self._emit_event(SimulationEvent(
                event_type=EventType.CYCLE_TICK,
                cycle=self.cycle_count,
                data=self.get_state()
            ))
            return
        
        if self.stall_next_cycle:
            self._handle_stall()
            self.stall_next_cycle = False
            # Stall means no advancement this cycle
            self._emit_event(SimulationEvent(
                event_type=EventType.CYCLE_TICK,
                cycle=self.cycle_count,
                data=self.get_state()
            ))
            return
        
        # Advance each pipeline
        for pipeline_id in range(self.num_pipelines):
            self._advance_pipeline(pipeline_id)
        
        # Emit cycle tick event with full state
        self._emit_event(SimulationEvent(
            event_type=EventType.CYCLE_TICK,
            cycle=self.cycle_count,
            data=self.get_state()
        ))
        
        # Check if simulation is complete
        if self._is_complete():
            self.running = False
            self._emit_event(SimulationEvent(
                event_type=EventType.SIMULATION_COMPLETE,
                cycle=self.cycle_count,
                data=self.stats.to_dict()
            ))
    
    def get_state(self) -> Dict:
        """
        Get the current state of the simulation.
        
        Returns:
            Dictionary containing full simulation state
        """
        return {
            'cycle': self.cycle_count,
            'running': self.running,
            'paused': self.paused,
            'pipelines': [
                [instr.to_dict() if instr else None for instr in pipeline]
                for pipeline in self.pipelines
            ],
            'instructions_remaining': len(self.instruction_queue),
            'statistics': self.stats.to_dict()
        }
    
    async def run_async(self):
        """
        Run the simulation asynchronously.
        This allows integration with async web frameworks.
        """
        while self.running and not self.paused:
            self.tick()
            await asyncio.sleep(self.clock_speed)
            
            if self._is_complete():
                self.running = False
                break
    
    def run_sync(self):
        """
        Run the simulation synchronously in a blocking manner.
        Useful for testing or standalone execution.
        """
        while self.running and not self.paused:
            self.tick()
            time.sleep(self.clock_speed)
            
            if self._is_complete():
                self.running = False
                break


# ========== WebSocket/API Integration Helper ==========

class PipelineOrchestra:
    """
    High-level interface for integrating with web frameworks.
    Provides async/await support for WebSocket or REST API integration.
    """
    
    def __init__(self, pipeline_type: PipelineType = PipelineType.CLASSIC):
        self.simulator = PipelineSimulator(
            pipeline_type=pipeline_type,
            event_callback=self._handle_event
        )
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_listeners: List[Callable] = []
    
    def _handle_event(self, event: SimulationEvent):
        """Internal event handler that queues events"""
        # For async: put in queue
        try:
            self.event_queue.put_nowait(event)
        except:
            pass
        
        # For sync: call all listeners
        for listener in self.event_listeners:
            try:
                listener(event)
            except:
                pass
    
    def add_event_listener(self, callback: Callable[[SimulationEvent], None]):
        """Add a synchronous event listener"""
        self.event_listeners.append(callback)
    
    async def get_next_event(self) -> SimulationEvent:
        """Get the next event from the queue (async)"""
        return await self.event_queue.get()
    
    async def start_simulation(self, melody: str = 'happy_birthday', speed: float = 0.3):
        """Start the simulation asynchronously"""
        self.simulator.set_clock_speed(speed)
        self.simulator.start(melody)
        asyncio.create_task(self.simulator.run_async())
    
    def control_command(self, command: str, **kwargs):
        """
        Execute a control command.
        
        Args:
            command: Command to execute ('start', 'pause', 'reset', 'stall', 'flush')
            **kwargs: Additional arguments for the command
        """
        if command == 'start':
            melody = kwargs.get('melody', 'happy_birthday')
            speed = kwargs.get('speed', 0.3)
            self.simulator.set_clock_speed(speed)
            self.simulator.start(melody)
        elif command == 'pause':
            self.simulator.pause()
        elif command == 'reset':
            self.simulator.reset()
        elif command == 'stall':
            self.simulator.inject_stall()
        elif command == 'flush':
            self.simulator.inject_flush()
        elif command == 'set_speed':
            speed = kwargs.get('speed', 0.3)
            self.simulator.set_clock_speed(speed)
    
    def get_state(self) -> Dict:
        """Get current simulation state"""
        return self.simulator.get_state()


# ========== Example Usage & Testing ==========

def example_event_handler(event: SimulationEvent):
    """Example event handler that prints events"""
    if event.event_type == EventType.NOTE_PLAY:
        print(f"🎵 Cycle {event.cycle}: Play note {event.data['note']}")
    elif event.event_type == EventType.INSTRUCTION_COMPLETED:
        instr = event.data['instruction']
        print(f"✓ Cycle {event.cycle}: Completed instruction '{instr['name']}'")
    elif event.event_type == EventType.STALL_INJECTED:
        print(f"⚠️  Cycle {event.cycle}: STALL injected")
    elif event.event_type == EventType.FLUSH_INJECTED:
        print(f"💥 Cycle {event.cycle}: FLUSH injected")
    elif event.event_type == EventType.SIMULATION_COMPLETE:
        print(f"\n🏁 Simulation complete!")
        print(f"Statistics: {json.dumps(event.data, indent=2)}")


def run_demo():
    """Run a demonstration of the pipeline simulator"""
    print("=" * 60)
    print("Interactive Pipeline Orchestra - Backend Demo")
    print("=" * 60)
    print()
    
    # Create simulator with event handler
    simulator = PipelineSimulator(
        pipeline_type=PipelineType.CLASSIC,
        clock_speed=0.2,  # Faster for demo
        event_callback=example_event_handler
    )
    
    # Start simulation
    print("Starting simulation with 'Happy Birthday' melody...")
    print()
    simulator.start('happy_birthday')
    
    # Run for a few cycles, then inject a stall
    for i in range(3):
        simulator.tick()
    
    print("\nInjecting stall...")
    simulator.inject_stall()
    simulator.tick()
    
    # Continue running
    print("\nContinuing simulation...\n")
    simulator.run_sync()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run the demonstration
    run_demo()
    
    print("\n\nTo integrate with a web server:")
    print("1. Use PipelineOrchestra class for async/await support")
    print("2. Connect event_queue to WebSocket for real-time updates")
    print("3. Expose control_command() via REST API endpoints")
    print("4. Example WebSocket integration:")
    print("""
    async def websocket_handler(websocket):
        orchestra = PipelineOrchestra()
        
        # Send events to client
        async def send_events():
            while True:
                event = await orchestra.get_next_event()
                await websocket.send(json.dumps(event.to_dict()))
        
        # Start event sender
        asyncio.create_task(send_events())
        
        # Handle commands from client
        async for message in websocket:
            data = json.loads(message)
            orchestra.control_command(data['command'], **data.get('params', {}))
    """)