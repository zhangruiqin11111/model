package test;
import java.net.Socket;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
public class client {
	public static void main(String[] args) throws IOException {
			String HOST = "127.0.0.1";//Ҫ���ӵķ����IP��ַ
	        int PORT = 12345;//Ҫ���ӵķ���˶�Ӧ�ļ����˿�
	        Socket socket = new Socket(HOST, PORT);
	        OutputStream outputStream = socket.getOutputStream();
	        outputStream.write(("Hello server with java").getBytes());
	        outputStream.flush();
	        System.out.println(socket);
	        InputStream is = socket.getInputStream();
	        byte[] bytes = new byte[1024];
	        int n = is.read(bytes);
	        System.out.println(new String(bytes, 0, n));
	        is.close();
	        socket.close();
	    }
}
