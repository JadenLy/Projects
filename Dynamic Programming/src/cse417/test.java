package cse417;

import java.util.*;

public class test {
	public static void main(String[] args) throws Exception {
		List<Integer> test = new ArrayList<Integer>();
		test.add(1);
		test.add(2);
		test.add(3);
		
		for (int i: test) {
			if (test.contains(1)){
				test.remove(1);
			}
			System.out.println(i);
		}
		
	}
}
