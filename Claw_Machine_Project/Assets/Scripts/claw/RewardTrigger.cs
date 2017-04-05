using UnityEngine;
using System.Collections;
using MLPlayer;

public class RewardTrigger : MonoBehaviour {

	public MyAgent agent;
	public ClawController claw;

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
	
	}

	void OnTriggerEnter(Collider other) {
		if (claw.dropped) {
			print ("reward");
			agent.state.reward += 1;
		}
	}
}
