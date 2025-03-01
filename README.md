# Vehicle Routing Problem with Time Windows and Capacity Constraints

## Problem Description

The following paragraphs describe the problem constraints and objectives in detail. These details will be crucial to your implementation, so please read them carefully.

A set of **nurses** (equivalent to vehicles) cares for a set of **patients** (equivalent to customers). There is **one depot**, and each patient must be visited **exactly once**. Each nurse starts at the depot, visits an arbitrary subset of the patients to provide care, and returns to the depot. This is called a **route**. Each nurse must return to the depot before the specified return time.

Each nurse has a **capacity**, and each patient has a **demand**. The total demand of the patients in a nurse’s route must not exceed their capacity. All nurses in a given problem instance have the **same capacity**, but the patients’ demands **vary**. The demand represents the strain on the nurse to perform patient care, while the nurse’s capacity represents how much strain they can handle during a route.

Each patient has:
- **A care time** – the time required to care for the patient.
- **A time window** – a start and end time defining when care can occur.

The **time windows are strict**:
- The **start time** defines the earliest time a nurse can begin care.
- The **end time** defines the latest time by which care must be completed.
- If a nurse arrives before the start time, they must **wait** until the start time before beginning care.

### Constraints
The problem is subject to the following constraints:

1. Each route starts at the depot at **time 0**.
2. Each route **ends at the depot** and must arrive before the given depot return time.
3. The total demand on a route must be **≤ the nurse’s capacity**.
4. Each patient visit must occur **within the respective time window**.
5. Each patient is visited **exactly once**.

### Objective
The goal is to **minimize the total travel time**, i.e., the sum of the travel time of all routes.  
**Note:** This does **not** include care time or potential waiting time.