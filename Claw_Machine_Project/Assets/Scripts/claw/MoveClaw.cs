using UnityEngine;
using System.Collections;

public class MoveClaw : MonoBehaviour {

	public Transform leftFinger, rightFinger, body;
	public bool open = true;
	Rigidbody bodyRB;
	public Vector3 target;

	//public string command = "";

	// Use this for initialization
	void Start () {
		bodyRB = body.GetComponent<Rigidbody> ();
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	public void SetOpen(bool open)
	{
		this.open = open;
	}

	//void FixedUpdate() {

	// 0 = dontMove
	// 1 = move
	// 2 = moveTo
	public void Move(float x, float y, float z, int moveMode, bool open) {
		//print (x + " " + y + " " + z + " " + moveMode + " " + open);

		this.open = open;
		Vector3 move = Vector3.zero;
		float leftF = 0;
		float rightF = 0;

		bodyRB.constraints = RigidbodyConstraints.FreezeAll;

		if (moveMode == 2) {
			move = (new Vector3(x, y, z) - bodyRB.position);
			//move.y = 0;
			float a = 3;

			//if (move.magnitude > a) {
				move.Normalize ();
				move = move * a;
			//}

		} else if (moveMode == 1) {
			move = new Vector3 (x, y, z);
		}

		if (Mathf.Abs (move.x) > 0) {
			bodyRB.constraints &= ~RigidbodyConstraints.FreezePositionX;
		}
		if (Mathf.Abs (move.y) > 0) {
			bodyRB.constraints &= ~RigidbodyConstraints.FreezePositionY;
		}
		if (Mathf.Abs (move.z) > 0) {
			bodyRB.constraints &= ~RigidbodyConstraints.FreezePositionZ;
		}
			
		if (!open) {
			leftF = 100;
			rightF = 100;
		} else if (open) {
			leftF = -100;
			rightF = -100;
		}

		HingeJoint hingeL = leftFinger.GetComponent<HingeJoint> ();
		HingeJoint hingeR = rightFinger.GetComponent<HingeJoint> ();
		JointMotor motorL = hingeL.motor;
		JointMotor motorR = hingeR.motor;
		motorL.force = 300;
		motorR.force = 300;
		motorL.targetVelocity = leftF;
		motorR.targetVelocity = (hingeL.angle - hingeR.angle) * 100f;//rightF;
		motorL.freeSpin = false;
		motorR.freeSpin = false;

		//hingeL.angle = hingeR.angle;
		//if (move.magnitude > 0) {
		body.GetComponent<Rigidbody> ().velocity = move;
		//} else {
		//}
		//GetComponent<Rigidbody> ().AddForce (move, ForceMode.Impulse);

		hingeL.motor = motorL;
		hingeR.motor = motorR;
		hingeL.useMotor = true;
		hingeR.useMotor = true;

		//Debug.Log (command);
	}


}
