using UnityEngine;
using System.Collections;
using MLPlayer;

public class AgentTouch : MonoBehaviour {

	public int index;
	public MyAgent agent;

	// Use this for initialization
	void Start () {
	
	}

	// Update is called once per frame
	void Update () {
	}

	void FixedUpdate() {
		//agent.GetTouch()[index] = 0.0f;
	}

	void OnTriggerEnter(Collider other) {
		agent.GetTouch () [index] = 1;//info.impulse.magnitude;
		//print ("sensor touch");
		//agent.state.reward += 0.1f;
	}

	void OnTriggerStay(Collider other) {
		agent.GetTouch () [index] = 1;
	}
}
