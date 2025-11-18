# Memory Exercise: Dining Assistant (Starter)

In this exercise, you'll build a dining assistant that remembers each user's dietary preferences and personalizes its responses across multiple conversation sessions.

## Part 1: Conceptual Questions

1. **The role of checkpointers in persisting state**

2. **The difference between state and config**

3. **How thread isolation works in LangGraph**

## Part 2: Implement the Dining Assistant

Follow the TODOs in the code cell below to complete the dining assistant. Be sure to:

- Import `InMemorySaver` from `langgraph.checkpoint.memory` and compile the workflow with it.
- Expand `MemoryState` to include a `user_memory` dictionary to store dietary preferences and visit counts (user_memory: {"diet": List[str], "visits": int})
- Write a node function `remember_preferences(state: MemoryState) -> MemoryState` that:
  - Detects dietary preferences (vegan, vegetarian, gluten-free) from the latest user message
  - Stores them in `state["user_memory"]["diet"]` (a list)
  - Increments a visit counter in `state["user_memory"]["visits"]`
  - Appends a personalized dish suggestion and a welcome-back message on return visits
- Add this new node to the workflow after the greet node and update the edges accordingly.
- Compile the workflow with an `InMemorySaver` to enable persistence.
- Test your implementation by invoking the workflow multiple times with the same `thread_id`.


## Coding Exercise: Implementing Persistent Memory

This hands-on exercise lets you apply what you've learned. You'll extend the LangGraph application below to incorporate memory and personalize responses.

**Scenario:** You are building a simple dining assistant that remembers each user's dietary preferences and personalizes its responses.
.

### Tasks to Complete

Modify the application below by implementing the following:

1. **Add Memory Persistence**
   - Import `InMemorySaver` from `langgraph.checkpoint.memory`
   - Modify the workflow to compile with this checkpointer

2. **Track User Preferences**
    - Use `InMemorySaver` for state persistence so that the assistant remembers preferences across separate invocations.
    - Pass a unique thread ID in the `config` to ensure conversation isolation for each user.
    - Extend the application state with a `user_memory` dictionary to store dietary preferences (e.g. vegetarian, vegan, gluten-free) and a visit counter.
    - Implement the `remember_preferences` function that detects preferences from incoming messages, updates the stored preferences and visit count, and appends personalized suggestions and welcome-back messages.
    - Connect the `remember_preferences` node to the appropriate edges in the workflow.
    - Compile the workflow with a checkpointer and test it by invoking it multiple times with the same thread ID to verify persistence.

3. **Test Persistence**
   - Invoke the workflow twice using the same `thread_id`
   - Observe how the second invocation uses the stored preference