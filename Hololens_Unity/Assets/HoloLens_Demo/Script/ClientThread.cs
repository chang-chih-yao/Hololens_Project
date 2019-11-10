using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;

class ClientThread
{
    public struct Struct_Internet
    {
        public string ip;
        public int port;
    }

    private Socket clientSocket;//連線使用的Socket
    private Struct_Internet internet;
    public string receiveMessage;
    static public string receive_joint;
    private string sendMessage;


    public int recieve_flag;
    public byte[] frame_img;
    public Boolean begin_flag;
    public Boolean rec;
    public long dataLength;
    public long dataLength_frame;

    private Thread threadReceive;
    private Thread threadConnect;

    public ClientThread(AddressFamily family, SocketType socketType, ProtocolType protocolType, string ip, int port)
    {
        clientSocket = new Socket(family, socketType, protocolType);
        internet.ip = ip;
        internet.port = port;
        receiveMessage = null;
        receive_joint = null;
        recieve_flag = 0;
        begin_flag = true;
        rec = true;
    }

    public void StartConnect()
    {
        threadConnect = new Thread(Accept);
        threadConnect.Start();
    }

    public void StopConnect()
    {
        try
        {
            clientSocket.Close();
        }
        catch (Exception)
        {

        }
    }

    public void Send(string message)
    {
        if (message == null)
            throw new NullReferenceException("message不可為Null");
        else
            sendMessage = message;
        SendMessage();
    }

    public void Receive()
    {
        if (threadReceive != null && threadReceive.IsAlive == true)
            return;
        threadReceive = new Thread(ReceiveMessage);
        threadReceive.IsBackground = true;
        threadReceive.Start();
    }

    private void Accept()
    {
        try
        {
            clientSocket.Connect(IPAddress.Parse(internet.ip), internet.port);//等待連線，若未連線則會停在這行
        }
        catch (Exception)
        {
        }
    }

    private void SendMessage()
    {
        try
        {
            if (clientSocket.Connected == true)
            {
                clientSocket.Send(Encoding.ASCII.GetBytes(sendMessage));
            }
        }
        catch (Exception)
        {

        }
    }

    private void ReceiveMessage()
    {
        if (begin_flag)
        {
            if (clientSocket.Connected == true)
            {
                try
                {
                    clientSocket.Send(Encoding.ASCII.GetBytes("Unity_demo"));
                }
                catch (Exception)
                {

                }

                byte[] bytes = new byte[1024];
                long dataLength = clientSocket.Receive(bytes);
                receiveMessage = Encoding.ASCII.GetString(bytes);
                begin_flag = false;
                rec = false;
            }
        }
        else
        {
            if (clientSocket.Connected == true)
            {
                rec = true;
                try
                {
                    clientSocket.Send(Encoding.ASCII.GetBytes("echo"));
                }
                catch (Exception)
                {

                }

                byte[] bytes = new byte[4];
                dataLength = clientSocket.Receive(bytes);
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(bytes);
                int frame_size = BitConverter.ToInt32(bytes, 0);
                frame_img = new byte[frame_size];
                dataLength_frame = clientSocket.Receive(frame_img);

                byte[] co_str = new byte[1024];
                dataLength = clientSocket.Receive(co_str);
                receive_joint = Encoding.ASCII.GetString(co_str);

                recieve_flag = 1;

                //Mat mat_img = new Mat(1, frame_img.Length, CvType.CV_8U);
                //mat_img.put(0, 0, frame_img);
                //frame_img_mat = Imgcodecs.imdecode(mat_img, 1);
                //receiveMessage = Encoding.ASCII.GetString(bytes);
            }
        }

    }
}