using UnityEngine;
//using UnityEditor;
using System.Collections.Generic;
using System.Threading;

namespace MLPlayer
{
	public class SceneController : MonoBehaviour
	{
		//singleton
		protected static SceneController instance;

		public static SceneController Instance {
			get {
				if (instance == null) {
					instance = (SceneController)FindObjectOfType (typeof(SceneController));
					if (instance == null) {
						Debug.LogError ("An instance of" + typeof(SceneController) + "is needed in the scene,but there is none.");
					}
				}
				return instance;
			}
		}
			
		private static int msgEvery = 10;
		private static int renderEvery = 10;
		private int episodeFrameLength = 10000;

		[SerializeField] int fps = 200;

		[SerializeField] public Agent agent;
		public static AIServer server;
		public static bool FinishFlag = false;
		public static bool newMessage = false;
		public bool useServer = true;

		[SerializeField] Environment environment;
		private static int frame = 0;
		private static int lastMessage = 1000000;
		public static ManualResetEvent received = new ManualResetEvent (false);

		void Awake() {
			Debug.Log ("Starting...");
			QualitySettings.vSyncCount = 0;
			Application.targetFrameRate = fps;

			string[] arguments = System.Environment.GetCommandLineArgs ();

			if (arguments.Length >= 4) {
				renderEvery = int.Parse (arguments [2]);
				msgEvery = int.Parse (arguments [3]);
				useServer = int.Parse (arguments [4]) != 0;
			}

			try {
				MyAgent.objectCode = arguments[5];
			} catch {
			}

			Time.timeScale = renderEvery;
		}

		void Start ()
		{
			print ("starting server");
			QualitySettings.vSyncCount = 0;

			if (useServer) {
				server = new AIServer (agent);
			}

			StartNewEpisode ();
		}

		public void TimeOver ()
		{
			agent.EndEpisode ();
		}

		public void StartNewEpisode ()
		{
			Debug.Log ("new episode");
			frame = 0;
			if (lastMessage < 100000) {
				lastMessage = 0;
			}
			environment.OnReset ();
			agent.StartEpisode ();
		}

		public void Update() {
			
		}

		public void FixedUpdate ()
		{
			CustomUpdate ();
		}

		public static void NewMessage() {
			lastMessage = frame;
			//print ("new message: " + frame);// + " " + ((MyAgent)agent).mAction.moveX + " " + ((MyAgent)agent).mAction.moveZ);
		}
			
		public void CustomUpdate() {
			//print (FinishFlag);
			//print (server.agentMessage);

			if (!useServer) {
				if (agent.state.endEpisode) {
					StartNewEpisode ();
				}
				agent.ResetState ();

				return;
			}



			if (FinishFlag == false) {

				if ((server.agentMessage == null && frame - lastMessage >= msgEvery && agent.messageEnabled) || agent.state.endEpisode) {
					print (" last msg: " + lastMessage);
					agent.UpdateState ();
					server.PushAgentState (agent.state);

					if (agent.state.endEpisode) {
						StartNewEpisode ();
					}

					agent.ResetState ();

					received.Reset ();
					received.WaitOne ();
				}

				if (lastMessage > 100000) {
					received.Reset ();
					received.WaitOne ();
				}


				frame++;
					
				if (episodeFrameLength > 0 && frame > episodeFrameLength) {
					TimeOver ();
				}

			} else {
				//EditorApplication.isPlaying = false;
			}
		}
	}
}