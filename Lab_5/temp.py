from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.set_margins(10, 10, 10)  # Set margins to 10mm on all sides
pdf.set_auto_page_break(auto=True, margin=10)  # Enable automatic page breaks

lines = [
    "Enhanced Summary of ARQ Protocol Discussion",
    "1. Finite Sequence Number Space & Wraparound:",
    "   - With k bits, sequence numbers range from 0 to 2^k - 1.",
    "   - Frames are sent continuously, and sequence numbers wrap around to 0.",
    "   - This wraparound may cause confusion if old retransmitted frames (due to lost frames or acks) reappear as new ones.",
    "   - This can be the case when sender window is too large (2^k), and when all acks get lost, the receiver expects new sequence numbers from 0 to 2^k-1.",
    "   - If the sender is sending old frames, this can be an issue.",
    "",
    "2. Go-Back-N ARQ:",
    "   - To resolve this in Go-Back-N, we keep the sender's window size to be 2^k - 1 so that the receiver is waiting for the 2^k-th frame.",
    "   - Now, even if the sender repeats those 2^k -1 packets, the receiver rejects them all.",
    "",
    "3. Selective Repeat (SACK) ARQ:",
    "   - Both sender and receiver maintain a window of 2^k - 1.",
    "   - The receiver buffers out-of-order frames.",
    "   - Without extra information (like timestamps), delayed retransmissions may be mistaken as new if sequence numbers wrap.",
    "   - If all acks are lost, the receiver moves its window, and since it is Selective Repeat, it buffers those old frames as the next sequence number frames.",
    "   - For example, if the sender sends [0---2^k-2] and the receiver gets them all but loses all acks...",
    "   - Then the receiver's window shifts to [2^k-1, 0, 1, 2, ...], and if the sender retransmits [0---2^k-1],",
    "     they can be misinterpreted as new frames.",
    "",
    "4. Timestamping:",
    "   - Timestamps can differentiate old frames from new ones.",
    "   - They add extra overhead and require synchronized clocks.",
    "   - Traditional ARQ protocols avoid timestamps for simplicity.",
    "",
    "5. Window Size and Wraparound Issues:",
    "   - Windows sized 2^k - 1 or even 2^k - 2 risk ambiguity if they span the wraparound point.",
    "   - The receiver sees sequence numbers modulo 2^k, so retransmitted frames may be misinterpreted as new.",
    "",
    "6. Professor's Suggested Solution:",
    "   - Limiting the window to 2^(k-1) (half the sequence number space) prevents the active window from spanning the wraparound.",
    "   - This ensures retransmitted frames from an old cycle don't fall into the current window of new frames.",
    "",
    "Conclusion:",
    "   - Proper window sizing in ARQ protocols is crucial to avoid ambiguity.",
    "   - A smaller window (e.g., 2^(k-1)) guarantees a clear separation between cycles, ensuring reliable data interpretation.",
]

for line in lines:
    pdf.multi_cell(0, 8, line)  # multi_cell wraps text automatically
    pdf.ln(1)  # Add slight spacing between lines

pdf_file_path = "Summary.pdf"
pdf.output(pdf_file_path)
print("PDF file created:", pdf_file_path)
