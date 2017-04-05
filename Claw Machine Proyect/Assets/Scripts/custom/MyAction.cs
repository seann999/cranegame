using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace MLPlayer {
	public class MyAction : Action {
		//public float moveX = 0;
		//public float moveZ = 0;
		//public float claw = 0;
		char[] delimiters = {' '};
		public string[] tokens;
		public string command;

		public override void CommandReceived(string command) {
			//Debug.Log ("set " + command);
			tokens = command.Split (delimiters);
			this.command = tokens [0];
			//moveX = float.Parse (tokens[0]);
			//moveZ = float.Parse (tokens[1]);
			//claw = float.Parse (tokens[2]);
		}
	}
}
