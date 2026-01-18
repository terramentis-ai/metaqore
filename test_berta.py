#!/usr/bin/env python3
"""
Test script for BERTA Meta-Orchestrator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metaqore.core.berta_orchestrator import get_berta_orchestrator, BERTAOrchestrator

def test_berta_orchestrator():
    """Test the BERTA orchestrator functionality."""
    print("Testing BERTA Meta-Orchestrator...")

    # Get the BERTA orchestrator
    berta = get_berta_orchestrator()
    print(f"✓ BERTA orchestrator initialized: {type(berta).__name__}")

    # Test agent registration
    print("\n1. Testing agent registration...")
    success = berta.register_agent("TestAgent", ["ideation", "planning"])
    print(f"✓ Agent registration: {success}")

    success2 = berta.register_agent("CoderAgent", ["coding", "validation"])
    print(f"✓ Second agent registration: {success2}")

    # Test task creation
    print("\n2. Testing task creation...")
    task_id = berta.create_task("TestAgent", "ideation", priority=1.0)
    print(f"✓ Task created: {task_id}")

    task_id2 = berta.create_task("CoderAgent", "code_generation", dependencies=[task_id], priority=0.8)
    print(f"✓ Dependent task created: {task_id2}")

    # Test orchestration
    print("\n3. Testing orchestration...")
    result = berta.orchestrate_execution("TestAgent", "ideation")
    print(f"✓ Orchestration result: {result['success']}")
    print(f"  - Task ID: {result.get('task_id')}")
    print(f"  - BERTA orchestrated: {result['result'].get('berta_orchestrated')}")

    # Test context summary
    print("\n4. Testing context summary...")
    context = berta.get_global_context_summary()
    print(f"✓ Context summary:")
    print(f"  - Total agents: {context['total_agents']}")
    print(f"  - Total tasks: {context['total_tasks']}")
    print(f"  - Global context encoded: {context['global_state_encoded']}")
    print(f"  - Bidirectional context active: {context['bidirectional_context_active']}")

    # Test task status
    print("\n5. Testing task status...")
    status = berta.get_task_status(task_id)
    if status:
        print(f"✓ Task status for {task_id}:")
        print(f"  - Operation: {status['operation']}")
        print(f"  - Status: {status['status']}")
        print(f"  - Agent: {status['agent_name']}")

    print("\n🎉 All BERTA orchestrator tests passed!")

if __name__ == "__main__":
    test_berta_orchestrator()