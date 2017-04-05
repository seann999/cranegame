using UnityEngine;
using System.Collections;
using MLPlayer;

public class ClawController : MonoBehaviour {

	public MoveClaw claw;
	bool takeover = false;
	int frame = 0;
	//public Vector3 moveVec = new Vector3();
	//bool toggleClaw;
	public MyAgent agent;
	public bool dropped = false;
	int phase = 0;
	int countdown = -1;

	// Use this for initialization
	void Start () {
	
	}

	public void Reset() {
		takeover = false;//SceneController.server != null;
		frame = -1;
		phase = 0;
		//moveVec = new Vector3 (-1, -1, -1);
		claw.SetOpen (true);
		//claw.command = "";
		countdown = -1;
	}

	public int GetPhase() {
		return phase;
	}

	// Update is called once per frame
	void FixedUpdate () {
		MyAction act = agent.mAction;
		bool autoGrab = act.command == "autograb" || act.command == "auto";

		if (takeover) {
			if (countdown == 0 || frame == 0) {
				phase += 1;
				frame = 10000;
			}
			countdown--;
			frame--;

			if (phase == 0) {
				if (claw.target.x == -1 && claw.target.z == -1) {
					// nothing
				} else {
					//claw.target = new Vector3 (moveVec.x, claw.transform.position.y, moveVec.z);
					countdown = 0;
					phase = 1;
				}
			} else if (phase == 1) {
				if (Vector3.Distance (claw.transform.position, claw.target) > 0.01f) {
					//claw.command = "target";
					claw.Move (claw.target.x, claw.transform.position.y, claw.target.z, 2, true);
					countdown = 1;
				} else if (autoGrab) {
					countdown = 0;
				}

				if (autoGrab) {
					agent.messageEnabled = false;
				}
			} else if (phase == 2) {
				if (countdown < 0) {
					countdown = 500;
				}
				
				if (claw.transform.position.y > 2) {
					claw.Move (0, -1, 0, 1, true);
					//claw.command = "down";
				} else {
					//claw.command = "";
					countdown = 0;
					frame = -1;
				}
			} else if (phase == 3) {
				if (countdown == -1) {
					
					//claw.command = "claw";
					countdown = 30;
				} else {
					claw.Move (0, 0, 0, 0, false);
					//claw.command = "";
				}
			} else if (phase == 4) {
				if (claw.transform.position.y < 9) {
					claw.Move (0, 1, 0, 1, false);
					//claw.command = "up";
					countdown = 1;
				}
			} else if (phase == 5) {
				if (claw.transform.position.x > -5) {
					claw.Move (-1, 0, 0, 1, false);
					//claw.command = "left";
					countdown = 1;
				} else if (claw.transform.position.z > -5.5) {
					claw.Move (0, 0, -1, 1, false);
					//claw.command = "backward";
					countdown = 1;
				} else {
					countdown = 0;
				}
			} else if (phase == 6) {
				if (countdown == -1) {
					claw.Move (0, 0, 0, 1, false);
					//claw.command = "";
					countdown = 5;
					print ("wait");
				}
			} else if (phase == 7) {
				if (countdown == -1) {
					print ("release");
					dropped = true;

					//claw.command = "claw";
					countdown = 150;
				} else {
					claw.Move (0, 0, 0, 1, true);
				}
			} else {
				//claw.command = "";
				takeover = false;
				agent.EndEpisode();
			}

			//Debug.Log (frame);
		} else {
			dropped = false;

			if (act.command == "auto") {
				agent.messageEnabled = false;
				claw.target = new Vector3 (float.Parse (act.tokens [1]), float.Parse (act.tokens [2]), float.Parse (act.tokens [3]));

				phase = 1;
				takeover = true;
			} else if (act.command == "moveTo") {
				claw.Move (float.Parse (act.tokens [1]), float.Parse (act.tokens [2]), float.Parse (act.tokens [3]), 2, true);
			} else if (act.command == "reset") {
				takeover = false;
				agent.EndEpisode();
			} else if (Input.GetKey (KeyCode.Space) || act.command == "autograb") {
				agent.messageEnabled = false;
				claw.target = new Vector3 (claw.transform.position.x, 0, claw.transform.position.z);

				phase = 1;
				countdown = -1;
				frame = 0;
				takeover = true;
			} else {
				Vector3 moveVec = Vector3.zero;

				if (act.command == "move") {
					moveVec = new Vector3 (float.Parse (act.tokens [1]), float.Parse (act.tokens [2]), float.Parse (act.tokens [3]));
				}

				if (Input.GetKey (KeyCode.W)) {
					moveVec += new Vector3 (0, 0, 1);
				} else if (Input.GetKey (KeyCode.A)) {
					moveVec += new Vector3 (-1, 0, 0);
				} else if (Input.GetKey (KeyCode.S)) {
					moveVec += new Vector3 (0, 0, -1);
				} else if (Input.GetKey (KeyCode.D)) {
					moveVec += new Vector3 (1, 0, 0);
				} else if (Input.GetKey (KeyCode.Z)) {
					moveVec += new Vector3 (0, 1, 0);
				} else if (Input.GetKey (KeyCode.X)) {
					moveVec += new Vector3 (0, -1, 0);
				}

				bool open = claw.open;

				if (Input.GetKeyDown (KeyCode.Q) || act.command == "toggleClaw") {
					act.command = "";
					open = !open;
				}

				claw.Move (moveVec.x, moveVec.y, moveVec.z, 1, open);
			}
		}
	}
}
