from crewai.flow.flow import Flow, start, listen, and_
from typing import Any


# define our custom flow
class MyConditionalFlow(Flow):
    
    # define the start node
    @start()
    def start_node(self) -> str:
        return "Flow Started"
    
    # define the person name node
    @listen(start_node)
    def set_person_name(self) -> str:
        name = "Aman"
        self.state["name"] = name
        return name
    
    # define the person age node
    @listen(start_node)
    def set_person_age(self) -> int:
        age = 25
        self.state["age"] = age
        return age
    
    # define the merge node --> and condition
    @listen(and_(set_person_name, set_person_age))
    def merge_data_node(self) -> dict[str, Any]:
        person_dict = {
            "name": self.state["name"],
            "age": self.state["age"]
        }
        
        return person_dict
    
if __name__ == "__main__":
    # create the flow
    flow = MyConditionalFlow()
    
    # kickoff the fflow
    flow_output = flow.kickoff()
    
    # plot the flow
    flow.plot("conditional_flow.html")
    
    # print the result
    print("Flow Output: ", flow_output)