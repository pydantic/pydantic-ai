Example of a multi-agent flow where one agent delegates work to another, then hands off control to a third agent.

Demonstrates:

* [agent delegation](../multi-agent-applications.md#agent-delegation)
* [programmatic agent hand-off](../multi-agent-applications.md#programmatic-agent-hand-off)
* [usage limits](../agent.md#usage-limits)

In this scenario, a group of agents work together to find the best flight for a user.

The control flow for this example can be summarised as follows:

```mermaid
graph TD
  START --> search_agent("search agent")
  search_agent --> extraction_agent("extraction agent")
  extraction_agent --> search_agent
  search_agent --> human_confirm("human confirm")
  human_confirm --> search_agent
  search_agent --> FAILED
  human_confirm --> find_seat_function("find seat function")
  find_seat_function --> human_seat_choice("human seat choice")
  human_seat_choice --> find_seat_agent("find seat agent")
  find_seat_agent --> find_seat_function
  find_seat_function --> buy_flights("buy flights")
  buy_flights --> SUCCESS
```

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.flight_booking
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/flight_booking.py"}```
