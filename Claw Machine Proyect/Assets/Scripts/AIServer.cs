using UnityEngine;
//using UnityEditor;
using System;
using System.Collections.Generic;
using WebSocketSharp;
using WebSocketSharp.Server;
using WebSocketSharp.Net;
using System.Threading;
using MsgPack;
using Random = UnityEngine.Random;

namespace MLPlayer
{
	public class AIServer : MonoBehaviour
	{
		private WebSocketServer wssv;

		[SerializeField] string domain;
		//[SerializeField] int port;                                          
		public byte[] agentMessage;
		public Agent agent;
		private MsgPack.CompiledPacker packer;

		public AIServer (Agent _agent)
		{
			agent = _agent;
			packer = new MsgPack.CompiledPacker ();
			agentMessage = null;
		}

		public class CommunicationGym : WebSocketBehavior
		{
			public Agent agent { set; get; }
			MsgPack.BoxingPacker packer = new MsgPack.BoxingPacker ();
			private bool SendFlag=true;

			protected override void OnMessage (MessageEventArgs e)
			{
				agent.action.Set ((Dictionary<System.Object,System.Object>)packer.Unpack (e.RawData));

				//SceneController.received.Set ();

				//send state data 
				Sendmessage();
			
			}

			protected override void OnOpen ()
			{
				Debug.Log ("Socket Open");
				//SceneController.received.Set ();
				//Sendmessage ();
			}

			protected override void OnClose(CloseEventArgs e)
			{
				SceneController.FinishFlag=true;
				SceneController.received.Set ();
				Application.Quit ();
			}
				
			private void Sendmessage(){
				SceneController.server.agentMessage = null;
				SceneController.NewMessage ();
				SceneController.received.Set ();

				SendFlag = true;
	
				//send state data 
				while (SendFlag == true) {
					if (SceneController.server.agentMessage != null) {
						byte[] data = SceneController.server.agentMessage;
						//print ("send");
						Send (data);
						SendFlag = false;
						//print ("===");
						//SceneController.server.agentMessage = null;
					}
				}
			}
		}

		CommunicationGym instantiate ()
		{
			CommunicationGym service = new CommunicationGym ();
			service.agent = agent;
			return service;
		}

		string GetUrl(string domain,int port){
			return "ws://" + domain + ":" + port.ToString ();
		}

		void Awake ()
		{
			string[] arguments = System.Environment.GetCommandLineArgs ();

			int port = 5000;

			if (arguments.Length >= 2) {
				port = int.Parse (arguments [1]);
			}

			Random.seed = port;

			Debug.Log (GetUrl (domain, port));
			wssv = new WebSocketServer (GetUrl (domain, port));
			wssv.AddWebSocketService<CommunicationGym> ("/CommunicationGym", instantiate);
			wssv.Start ();


			if (wssv.IsListening) {
				Debug.Log ("Listening on port " + wssv.Port + ", and providing WebSocket services:");
				foreach (var path in wssv.WebSocketServices.Paths)
					Debug.Log ("- " + path);
			}
		}

		public void PushAgentState (State s)
		{
			byte[] msg = packer.Pack (s);  
			SceneController.server.agentMessage = msg;
		}

		void OnApplicationQuit ()
		{
			wssv.Stop ();
			Debug.Log ("websocket server exiteed");
		}
	}
}
