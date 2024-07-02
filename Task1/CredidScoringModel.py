import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> count1 = new HashMap<>();
        for (int num : nums1) {
            count1.put(num, count1.getOrDefault(num, 0) + 1);
        }

        Map<Integer, Integer> count2 = new HashMap<>();
        for (int num : nums2) {
            count2.put(num, count2.getOrDefault(num, 0) + 1);
        }

        List<Integer> intersection = new ArrayList<>();
        for (int num : count1.keySet()) {
            if (count2.containsKey(num)) {
                int minCount = Math.min(count1.get(num), count2.get(num));
                for (int i = 0; i < minCount; i++) {
                    intersection.add(num);
                }
            }
        }

        int[] result = new int[intersection.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = intersection.get(i);
        }

        return result;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int[] nums1 = {1, 2, 2, 1};
        int[] nums2 = {2, 2};
        int[] result = solution.intersect(nums1, nums2);
        for (int num : result) {
            System.out.print(num + " ");
        }
        System.out.println();  // Output: [2, 2]

        nums1 = new int[]{4, 9, 5};
        nums2 = new int[]{9, 4, 9, 8, 4};
        result = solution.intersect(nums1, nums2);
        for (int num : result) {
            System.out.print(num + " ");
        }
        System.out.println();  // Output: [4, 9] (or [9, 4])
    }
}
