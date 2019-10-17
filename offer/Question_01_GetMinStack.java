package part_01_stackandquene;
import java.util.EmptyStackException;
import java.util.Stack;

//import chapter_1_stackandqueue.Problem_01_GetMinStack.MyStack1;
//2019-10-17 zrq学习练习 
public class Question_01_GetMinStack {
	//新建两个栈 存放原始数据的栈 和存放最小值的栈
	private Stack<Integer> datastack = new Stack<Integer>();
	private Stack<Integer> minstack = new Stack<Integer>();
	//构造方法
	public Question_01_GetMinStack(){
		this.datastack = datastack;
		this.minstack = minstack;
	}
	//入栈方法  datastack直接入栈；data< minstack栈顶元素则入栈，否则入栈minstack栈顶元素
	public void push(int data) {
		if(minstack.isEmpty()){
			minstack.push(data);
		}
		if (data<=getmin()){
			minstack.push(data);
		}else{
			int datatop=minstack.pop();
			minstack.push(datatop);
		}
		datastack.push(data);
		
	}
	//出栈方法    datastack直接出栈   minstack从栈顶出栈
	public int pop(int data) {
		
		if(datastack.isEmpty()){
			throw new RuntimeException("栈为空");
			
		}
		minstack.pop();
		return datastack.pop();
	}
	public int getmin(){
		if(minstack==null){
			System.out.println("zhanweikong");
		}
		return minstack.peek();		
	}
	public static void main(String[] args) {
		Question_01_GetMinStack stack1 = new Question_01_GetMinStack();
		stack1.push(3);
		System.out.println(stack1.getmin());
		stack1.push(4);
		System.out.println(stack1.getmin());
		stack1.push(1);
		System.out.println(stack1.getmin());
		System.out.println(stack1.pop(1));
		System.out.println(stack1.getmin());

		System.out.println("=============");

		Question_01_GetMinStack stack2 = new Question_01_GetMinStack();
		stack2.push(3);
		System.out.println(stack2.getmin());
		stack2.push(4);
		System.out.println(stack2.getmin());
		stack2.push(1);
		System.out.println(stack2.getmin());
		System.out.println(stack2.pop(1));
		System.out.println(stack2.getmin());
	}


}
