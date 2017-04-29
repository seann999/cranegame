using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace MLPlayer {
	[System.Serializable]
	public class ExtraInfo
	{
		public float[] coords;
		public float[] touch = new float[2];

		private List<Transform> things;

		public ExtraInfo(List<Transform> things) {
			coords = new float[things.Count * 3];
			this.things = things;
		}

		public void Update() {
			for (int i = 0; i < things.Count; i += 3) {
				Vector3 v = things [i].position;
				coords [i] = v.x;
				coords [i + 1] = v.y;
				coords [i + 2] = v.z;
			}
		}
	}

	public class MyAgent : Agent {
		public static string objectCode = "aaaaa";

		public MyAction mAction = null;
		public ClawController claw;
		public Transform clawBody;
		public int frame = 0;

		public List<Transform> instances;

		private List<Transform> things = new List<Transform>();
		private ExtraInfo extraInfo;

		// Use this for initialization
		void Start () {
			base.Start();
			base.action = new MyAction();
			mAction = (MyAction)base.action;

			foreach (char c in objectCode) {
				int code = (int)c - (int)'a';
				print (code);

				Object obj = Instantiate (instances [code], new Vector3 (0, 0, 0), Quaternion.identity);
				things.Add((Transform) obj);
			}

			extraInfo = new ExtraInfo (things);
		}

		public override void StartEpisode ()
		{
			clawBody.transform.position = new Vector3 (0, 0, 0);
			messageEnabled = true;

			//Random.seed = 0;
			foreach (Transform t in things) {
				bool valid = false;
				int tries = 0;

				while (!valid && tries < 20) {
					t.position = new Vector3 (Random.value * 10f - 5f, -2, Random.value * 10f - 5f);
					Collider[] checkResult = Physics.OverlapSphere(t.position, 2);

					if (checkResult.Length <= 2) {
						valid = true;
					}

					tries++;

					//print (checkResult.Length);
					//valid = true;
				}

				t.rotation = Random.rotation;
				t.GetComponent<Rigidbody> ().velocity = Vector3.zero;
			}

			extraInfo.touch [0] = 0;
			extraInfo.touch [1] = 0;

			frame = 0;
		}

		public float[] GetTouch() {
			return extraInfo.touch;
		}

		public override void ResetState() {
			base.ResetState ();
			extraInfo.touch [0] = 0;
			extraInfo.touch [1] = 0;
		}

		public override void EndEpisode ()
		{
			base.EndEpisode ();
			claw.Reset ();
		}

		public override void UpdateState() {
			base.UpdateState ();

			/*string data = "[";

			for (int i = 0; i < things.Count; i++) {
				Transform t = things [i];
				data += "[" + t.position.x + "," + t.position.y + "," + t.position.z + "]";

				if (i < things.Count - 1) {
					data += ",";
				}
			}

			data += "]";*/

			extraInfo.Update ();
			string data = JsonUtility.ToJson(extraInfo);
			state.extra = System.Text.Encoding.ASCII.GetBytes(data);
		}
	
		// Update is called once per frame
		void FixedUpdate () {
			//claw.moveVec = new Vector3 (mAction.moveX, -mAction.claw, mAction.moveZ);
			frame++;
		}
	}
}
